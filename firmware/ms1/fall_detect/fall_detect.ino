/*
  fall.ino (Full)
  - Arduino Nano 33 BLE Sense Rev2 (BMI270)
  - TFLite Micro (Harvard TinyMLx / TensorFlowLite.h wrapper)
  - Model input: [1, 6, 50, 1] (NHWC)
  - Window source: [50][6] = (T x C): AccX,AccY,AccZ,GyrX,GyrY,GyrZ

  Notes:
  - You MUST replace kMean/kStd with your training mean/std for best results.
  - You may need to match sensor units to training units (g vs m/s^2, deg/s vs rad/s).
*/

#include <Arduino.h>

// -------------------- IMU (Nano 33 BLE Sense Rev2) --------------------
#include <Arduino_BMI270_BMM150.h>  // provides IMU object

// -------------------- TFLite Micro --------------------
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// -------------------- Your model --------------------
#include "model_int8.h"  // contains model_int8_tflite + model_int8_tflite_len

// -------------------- Config --------------------
static const int kFallClass = 1;
static const int kT = 50;   // frames per window
static const int kC = 6;    // Ax,Ay,Az,Gx,Gy,Gz

// Sampling (Hz). Your model likely assumes a fixed rate.
// Choose a rate that matches training, e.g. 50 Hz or 100 Hz.
static const int kSampleHz = 50;
static const uint32_t kSamplePeriodMs = 1000 / kSampleHz;

// If your training used m/s^2 and deg/s, set these accordingly.
// The Arduino IMU library typically returns:
// - acceleration in g (or m/s^2 depending on library version)
// - gyro in deg/s
// You MUST verify by printing a few samples and comparing expected magnitude.
// Quick sanity: at rest, |acc| should be ~1 if it's "g", or ~9.81 if it's m/s^2.
static const float kAccScale = 1.0f;   // multiply accel by this
static const float kGyroScale = 1.0f;  // multiply gyro by this

// Replace with your real training normalization stats (per channel).
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

// -------------------- Window buffer --------------------
static float g_window[kT][kC];
static int g_idx = 0;
static bool g_window_full = false;

// -------------------- TFLite Micro globals --------------------
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Increase if AllocateTensors fails.
constexpr int kTensorArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// -------------------- Helpers --------------------
static inline float safe_std(float s) { return (s == 0.0f) ? 1.0f : s; }

static inline int8_t quantize_int8(float x, float scale, int zero_point) {
  int32_t q = (int32_t) lroundf(x / scale + (float)zero_point);
  if (q < -128) q = -128;
  if (q > 127)  q = 127;
  return (int8_t) q;
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

// Push one frame (6 values) into ring window
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

// Copy ring buffer into a linear (time-ordered) temp buffer:
// oldest -> newest. (So model sees correct time order.)
static void BuildOrderedWindow(float out[kT][kC]) {
  // If not full yet, just copy [0..g_idx)
  if (!g_window_full) {
    for (int t = 0; t < g_idx; ++t) {
      for (int c = 0; c < kC; ++c) out[t][c] = g_window[t][c];
    }
    // pad the rest with last frame
    for (int t = g_idx; t < kT; ++t) {
      for (int c = 0; c < kC; ++c) out[t][c] = g_window[g_idx > 0 ? g_idx - 1 : 0][c];
    }
    return;
  }

  // When full: oldest is g_idx (next write), newest is g_idx-1.
  int start = g_idx;
  for (int t = 0; t < kT; ++t) {
    int src = (start + t) % kT;
    for (int c = 0; c < kC; ++c) out[t][c] = g_window[src][c];
  }
}

// Fill input tensor from window (T x C) -> model layout [1,6,50,1] (NHWC)
// We do: normalize per channel, then place at [c][t][0]
static bool FillInputFromWindow(const float ordered[kT][kC]) {
  if (!input_tensor) return false;

  if (input_tensor->dims->size != 4) {
    error_reporter->Report("Bad input dims: %d", input_tensor->dims->size);
    return false;
  }

  const int b  = input_tensor->dims->data[0];
  const int h  = input_tensor->dims->data[1];  // 6
  const int w  = input_tensor->dims->data[2];  // 50
  const int ch = input_tensor->dims->data[3];  // 1

  if (b != 1 || h != kC || w != kT || ch != 1) {
    error_reporter->Report("Unexpected input shape: [%d,%d,%d,%d]", b, h, w, ch);
    return false;
  }

  const float in_scale = input_tensor->params.scale;
  const int   in_zero  = input_tensor->params.zero_point;

  if (input_tensor->type == kTfLiteInt8) {
    int8_t* in = input_tensor->data.int8;

    for (int c = 0; c < kC; ++c) {
      const float mean = kMean[c];
      const float stdv = safe_std(kStd[c]);
      for (int t = 0; t < kT; ++t) {
        float x = ordered[t][c];
        float z = (x - mean) / stdv;
        int8_t q = quantize_int8(z, in_scale, in_zero);

        // NHWC flatten: [1,6,50,1] -> index = (c*50 + t)
        in[c * kT + t] = q;
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
        float x = ordered[t][c];
        float z = (x - mean) / stdv;
        in[c * kT + t] = z;
      }
    }
    return true;
  }

  error_reporter->Report("Unsupported input type: %d", input_tensor->type);
  return false;
}

static int PredictFromCurrentWindow() {
  float ordered[kT][kC];
  BuildOrderedWindow(ordered);

  if (!FillInputFromWindow(ordered)) {
    error_reporter->Report("Failed to fill input");
    return -1;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return -1;
  }

  if (!output_tensor) return -1;

  // Usually output [1,2]
  const int out_elems =
      (output_tensor->type == kTfLiteInt8)    ? (output_tensor->bytes / (int)sizeof(int8_t)) :
      (output_tensor->type == kTfLiteFloat32) ? (output_tensor->bytes / (int)sizeof(float)) :
                                                output_tensor->bytes;

  if (output_tensor->type == kTfLiteInt8) {
    const int8_t* out = output_tensor->data.int8;
    return argmax_int8(out, out_elems);
  }

  if (output_tensor->type == kTfLiteFloat32) {
    const float* out = output_tensor->data.f;
    return argmax_float(out, out_elems);
  }

  error_reporter->Report("Unsupported output type: %d", output_tensor->type);
  return -1;
}

// -------------------- Setup / Loop --------------------
void setup() {
  Serial.begin(115200);
  delay(1200);
  Serial.println("TFLite Micro Fall Detection (IMU -> Window -> Infer)");

  // --- IMU init ---
  if (!IMU.begin()) {
    Serial.println("❌ IMU.begin() failed. Check board: Nano 33 BLE Sense Rev2.");
    while (1) delay(1000);
  }
  Serial.println("✅ IMU started.");

  // Optional: print IMU rates if supported
  // Serial.print("Acc rate: "); Serial.println(IMU.accelerationSampleRate());
  // Serial.print("Gyro rate: "); Serial.println(IMU.gyroscopeSampleRate());

  // --- Model init ---
  model = tflite::GetModel(model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("❌ Model schema mismatch. Expected ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(" got ");
    Serial.println(model->version());
    while (1) delay(1000);
  }

  // ✅ FIXED: no stray ')'
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ AllocateTensors failed. Try increasing kTensorArenaSize.");
    while (1) delay(1000);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("✅ Model loaded & tensors allocated.");
  Serial.print("Input type: "); Serial.println((int)input_tensor->type);
  Serial.print("Input dims: ");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    if (i != input_tensor->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();

  Serial.print("Output type: "); Serial.println((int)output_tensor->type);
  Serial.print("Output dims: ");
  for (int i = 0; i < output_tensor->dims->size; i++) {
    Serial.print(output_tensor->dims->data[i]);
    if (i != output_tensor->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();

  Serial.println("---- Start streaming IMU ----");
}

// reads one IMU sample (blocking-ish, but only when data ready)
static bool ReadIMUSample(float& ax, float& ay, float& az, float& gx, float& gy, float& gz) {
  // Ensure both sensors have data (best effort).
  bool hasA = IMU.accelerationAvailable();
  bool hasG = IMU.gyroscopeAvailable();
  if (!hasA || !hasG) return false;

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

void loop() {
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if (now - last_ms < kSamplePeriodMs) return;
  last_ms = now;

  float ax, ay, az, gx, gy, gz;
  if (!ReadIMUSample(ax, ay, az, gx, gy, gz)) {
    // no data yet
    return;
  }

  // Push into rolling window
  PushFrame(ax, ay, az, gx, gy, gz);

  // Optional: print raw sample occasionally
  // static int ctr = 0;
  // if ((ctr++ % 25) == 0) {
  //   Serial.print("RAW ax,ay,az,gx,gy,gz: ");
  //   Serial.print(ax); Serial.print(", ");
  //   Serial.print(ay); Serial.print(", ");
  //   Serial.print(az); Serial.print(", ");
  //   Serial.print(gx); Serial.print(", ");
  //   Serial.print(gy); Serial.print(", ");
  //   Serial.println(gz);
  // }

  // Run inference only after we have a full window
  if (!g_window_full) return;

  int pred = PredictFromCurrentWindow();
  if (pred < 0) {
    Serial.println("❌ Prediction failed.");
    return;
  }

  if (pred == kFallClass) {
    Serial.println("🚨 Prediction: FALL (class=1)");
  } else {
    Serial.println("✅ Prediction: NOT FALL (class=0)");
  }
}