#include "Arduino_BMI270_BMM150.h"
#include <Arduino_LSM9DS1.h>

#define WINDOW_SIZE      50
#define NUM_FEATURES     6
#define SAMPLE_PERIOD_US 10000    // 100 Hz

// ------------------------------------------------------------------
// GLOBAL normalization constants (replace with real values later)
// ------------------------------------------------------------------

// Accelerometer
float mean_ax = 0.0f, std_ax = 1.0f;
float mean_ay = 0.0f, std_ay = 1.0f;
float mean_az = 0.0f, std_az = 1.0f;

// Gyroscope
float mean_gx = 0.0f, std_gx = 1.0f;
float mean_gy = 0.0f, std_gy = 1.0f;
float mean_gz = 0.0f, std_gz = 1.0f;

// ------------------------------------------------------------------
// Smoothing (Exponential IIR)
// ------------------------------------------------------------------
const float ALPHA = 0.90f;   // Strong smoothing

float ax_f = 0, ay_f = 0, az_f = 0;
float gx_f = 0, gy_f = 0, gz_f = 0;

// Window buffers
float windowRaw[WINDOW_SIZE][NUM_FEATURES];
float windowNorm[WINDOW_SIZE][NUM_FEATURES];
int winIndex = 0;
bool windowReady = false;

// Drift-proof timing
unsigned long nextSampleTime = 0;

// ==================================================================
// Read IMU safely — DO NOT force reads
// Returns true if new data was actually received
// ==================================================================
bool readIMUSafe(float &ax, float &ay, float &az,
                 float &gx, float &gy, float &gz)
{
    bool newData = false;

    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        newData = true;
    }

    if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx, gy, gz);
        newData = true;
    }

    return newData;
}


// ==================================================================
// Smoothing filter
// ==================================================================
void smoothIMU(float ax, float ay, float az,
               float gx, float gy, float gz)
{
    ax_f = ALPHA * ax_f + (1 - ALPHA) * ax;
    ay_f = ALPHA * ay_f + (1 - ALPHA) * ay;
    az_f = ALPHA * az_f + (1 - ALPHA) * az;

    gx_f = ALPHA * gx_f + (1 - ALPHA) * gx;
    gy_f = ALPHA * gy_f + (1 - ALPHA) * gy;
    gz_f = ALPHA * gz_f + (1 - ALPHA) * gz;
}


// ==================================================================
// Add filtered sample to 50x6 window
// ==================================================================
void addSample()
{
    windowRaw[winIndex][0] = ax_f;
    windowRaw[winIndex][1] = ay_f;
    windowRaw[winIndex][2] = az_f;
    windowRaw[winIndex][3] = gx_f;
    windowRaw[winIndex][4] = gy_f;
    windowRaw[winIndex][5] = gz_f;

    winIndex++;

    if (winIndex >= WINDOW_SIZE) {
        winIndex = 0;
        windowReady = true;
    }
}


// ==================================================================
// Apply GLOBAL normalization (x_norm = (x - mean) / std)
// ==================================================================
void applyGlobalNorm()
{
    for (int t = 0; t < WINDOW_SIZE; t++) {
        windowNorm[t][0] = (windowRaw[t][0] - mean_ax) / std_ax;
        windowNorm[t][1] = (windowRaw[t][1] - mean_ay) / std_ay;
        windowNorm[t][2] = (windowRaw[t][2] - mean_az) / std_az;

        windowNorm[t][3] = (windowRaw[t][3] - mean_gx) / std_gx;
        windowNorm[t][4] = (windowRaw[t][4] - mean_gy) / std_gy;
        windowNorm[t][5] = (windowRaw[t][5] - mean_gz) / std_gz;
    }
}


// ==================================================================
// Print window
// ==================================================================
void printWindow()
{
    Serial.println("=== Normalized 50x6 Window (Smoothed) ===");

    for (int i = 0; i < WINDOW_SIZE; i++) {
        Serial.print("[");
        for (int j = 0; j < NUM_FEATURES; j++) {
            Serial.print(windowNorm[i][j], 4);
            if (j < NUM_FEATURES - 1) Serial.print(", ");
        }
        Serial.println("]");
    }
}


// ==================================================================
// Setup
// ==================================================================
void setup()
{
    Serial.begin(115200);

    if (!IMU.begin()) {
        Serial.println("IMU init failed");
        while (1);
    }

    Serial.println("Running: smoothing + safe reads + drift-proof timing");
    nextSampleTime = micros();
}


// ==================================================================
// Loop
// ==================================================================
void loop()
{
    unsigned long now = micros();

    // Drift-proof sampling interval
    if (now >= nextSampleTime) {
        nextSampleTime += SAMPLE_PERIOD_US;

        float ax, ay, az, gx, gy, gz;

        // Read only if the IMU provides new data
        bool gotNewData = readIMUSafe(ax, ay, az, gx, gy, gz);

        if (!gotNewData) {
            // No new data this cycle — skip processing
            return;
        }

        // Smooth
        smoothIMU(ax, ay, az, gx, gy, gz);

        // Add to window
        addSample();
    }

    // When a window is complete
    if (windowReady) {
        windowReady = false;

        applyGlobalNorm();
        printWindow();

        // Later: feed windowNorm[][] to ML model
    }
}
