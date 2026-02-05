/*
  fall.ino (Full) — NO Serial prints, NO delay()
  - Arduino Nano 33 BLE Sense Rev2 (BMI270)
  - TFLite Micro
  - Model input: [1, 6, 50, 1] (NHWC)
  - Window source: [50][6] = (T x C): AccX,AccY,AccZ,GyrX,GyrY,Gz

  Behavior:
  - When FALL is detected the FIRST time (NOT FALL -> FALL),
    it sends "FALL" via BLE notify once, and PAUSES sampling/inference.
  - BLE continues to work while paused (BLE.poll() still runs).
  - No resume command from app (paused remains until reset/power cycle).
*/

#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>
#include <ArduinoBLE.h>

/* ===================== BLE UUIDs ===================== */
#define SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define RX_UUID      "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define TX_UUID      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLEService vestService(SERVICE_UUID);

// Keep RX characteristic if you still want the app to be able to write (optional).
// If you truly don't need RX at all, you can delete rxChar + related adds.
BLEStringCharacteristic rxChar(RX_UUID, BLEWrite, 20);

BLEStringCharacteristic txChar(TX_UUID, BLENotify, 20);

/* ===================== Fall latch / pause ===================== */
static bool g_fall_latched = false;
static bool g_paused = false;

/* ===================== TFLite Micro ===================== */
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

/* ===================== Your model ===================== */
#include "model_int8.h"

/* ===================== Config ===================== */
static const int kFallClass = 1;
static const int kT = 50;
static const int kC = 6;

static const int kSampleHz = 50;
static const uint32_t kSamplePeriodMs = 1000 / kSampleHz;

static const float kAccScale  = 1.0f;
static const float kGyroScale = 1.0f;

static const float kMean[kC] = {
  1.4312255e-02f,
 -8.7299287e-01f,
  5.7684090e-03f,
 -1.2964634e+00f,
  8.8249311e+00f,
 -6.0928702e-02f
};

static const float kStd[kC] = {
  0.2970297f,
  0.43655434f,
  0.41131276f,
  38.82977f,
  47.30563f,
  23.103584f
};

/* ===================== Window buffer ===================== */
static float g_window[kT][kC];
static int g_idx = 0;
static bool g_window_full = false;

/* ===================== TFLite Micro globals ===================== */
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

constexpr int kTensorArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

/* ===================== Helpers ===================== */
static inline float safe_std(float s) { return (s == 0.0f) ? 1.0f : s; }

static inline int8_t quantize_int8(float x, float scale, int zero_point) {
  int32_t q = (int32_t)lroundf(x / scale + (float)zero_point);
  if (q < -128) q = -128;
  if (q > 127)  q = 127;
  return (int8_t)q;
}

static int argmax_int8(const int8_t* data, int n) {
  int best_i = 0;
  int best_v = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] > best_v) {
      best_v = data[i];
      best_i = i;
    }
  }
  return best_i;
}

static int argmax_float(const float* data, int n) {
  int best_i = 0;
  float best_v = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] > best_v) {
      best_v = data[i];
      best_i = i;
    }
  }
  return best_i;
}

static void PushFrame(float ax, float ay, float az, float gx, float gy, float gz) {
  g_window[g_idx][0] = ax;
  g_window[g_idx][1] = ay;
  g_window[g_idx][2] = az;
  g_window[g_idx][3] = gx;
  g_window[g_idx][4] = gy;
  g_window[g_idx][5] = gz;

  g_idx++;
  if (g_idx >= kT) {
    g_idx = 0;
    g_window_full = true;
  }
}

static void BuildOrderedWindow(float out[kT][kC]) {
  if (!g_window_full) {
    for (int t = 0; t < g_idx; ++t) {
      for (int c = 0; c < kC; ++c) out[t][c] = g_window[t][c];
    }
    for (int t = g_idx; t < kT; ++t) {
      for (int c = 0; c < kC; ++c) out[t][c] = g_window[g_idx > 0 ? g_idx - 1 : 0][c];
    }
    return;
  }

  int start = g_idx;
  for (int t = 0; t < kT; ++t) {
    int src = (start + t) % kT;
    for (int c = 0; c < kC; ++c) out[t][c] = g_window[src][c];
  }
}

static bool FillInputFromWindow(const float ordered[kT][kC]) {
  if (!input_tensor) return false;
  if (input_tensor->dims->size != 4) return false;

  const int b  = input_tensor->dims->data[0];
  const int h  = input_tensor->dims->data[1];
  const int w  = input_tensor->dims->data[2];
  const int ch = input_tensor->dims->data[3];

  if (b != 1 || h != kC || w != kT || ch != 1) return false;

  const float in_scale = input_tensor->params.scale;
  const int   in_zero  = input_tensor->params.zero_point;

  if (input_tensor->type == kTfLiteInt8) {
    int8_t* in = input_tensor->data.int8;
    for (int c = 0; c < kC; ++c) {
      const float mean = kMean[c];
      const float stdv = safe_std(kStd[c]);
      for (int t = 0; t < kT; ++t) {
        float z = (ordered[t][c] - mean) / stdv;
        in[c * kT + t] = quantize_int8(z, in_scale, in_zero);
      }
    }
    return true;
  }

  if (input_tensor->type == kTfLiteFloat32) {
    float* in = input_tensor->data.f;
    for (int c = 0; c < kC; ++c) {
      const float mean = kMean[c];
      const float stdv = safe_std(kStd[c]);
      for (int t = 0; t < kT; ++t) {
        in[c * kT + t] = (ordered[t][c] - mean) / stdv;
      }
    }
    return true;
  }

  return false;
}

static int PredictFromCurrentWindow() {
  float ordered[kT][kC];
  BuildOrderedWindow(ordered);

  if (!FillInputFromWindow(ordered)) return -1;
  if (interpreter->Invoke() != kTfLiteOk) return -1;
  if (!output_tensor) return -1;

  const int out_elems =
      (output_tensor->type == kTfLiteInt8)    ? (output_tensor->bytes / (int)sizeof(int8_t)) :
      (output_tensor->type == kTfLiteFloat32) ? (output_tensor->bytes / (int)sizeof(float)) :
                                                output_tensor->bytes;

  if (output_tensor->type == kTfLiteInt8) {
    return argmax_int8(output_tensor->data.int8, out_elems);
  }
  if (output_tensor->type == kTfLiteFloat32) {
    return argmax_float(output_tensor->data.f, out_elems);
  }
  return -1;
}

static bool ReadIMUSample(float& ax, float& ay, float& az, float& gx, float& gy, float& gz) {
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return false;

  float rax, ray, raz;
  float rgx, rgy, rgz;

  if (!IMU.readAcceleration(rax, ray, raz)) return false;
  if (!IMU.readGyroscope(rgx, rgy, rgz)) return false;

  ax = rax * kAccScale;
  ay = ray * kAccScale;
  az = raz * kAccScale;
  gx = rgx * kGyroScale;
  gy = rgy * kGyroScale;
  gz = rgz * kGyroScale;
  return true;
}

/* ===================== Setup / Loop ===================== */
void setup() {
  if (!IMU.begin()) {
    while (1) { /* stop */ }
  }

  if (!BLE.begin()) {
    while (1) { /* stop */ }
  }

  BLE.setLocalName("Vest");
  BLE.setAdvertisedService(vestService);

  vestService.addCharacteristic(rxChar); // optional
  vestService.addCharacteristic(txChar);

  BLE.addService(vestService);
  BLE.advertise();

  model = tflite::GetModel(model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    while (1) { BLE.poll(); }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    while (1) { BLE.poll(); }
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
}

void loop() {
  // BLE stays alive even when paused
  BLE.poll();

  // Optional: if you truly don't need RX at all, remove this entire block.
  // (No resume logic here anymore.)
  if (rxChar.written()) {
    String cmd = rxChar.value();
    cmd.trim();
    txChar.writeValue("ACK:" + cmd); // just acknowledge, no state changes
  }

  // If paused after fall, stop sampling/inference, but BLE still runs
  if (g_paused) return;

  // Fixed-rate sampling
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if ((uint32_t)(now - last_ms) < kSamplePeriodMs) return;
  last_ms = now;

  float ax, ay, az, gx, gy, gz;
  if (!ReadIMUSample(ax, ay, az, gx, gy, gz)) return;

  PushFrame(ax, ay, az, gx, gy, gz);
  if (!g_window_full) return;

  int pred = PredictFromCurrentWindow();
  if (pred < 0) return;

  if (pred == kFallClass && !g_fall_latched) {
    g_fall_latched = true;
    g_paused = true;
    txChar.writeValue("FALL");
  }
}