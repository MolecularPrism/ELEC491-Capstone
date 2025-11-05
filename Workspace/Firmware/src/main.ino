#include "imu.h"
#include "inflate.h"
#include "model_inference.h"

#define WINDOW_SIZE 100   // 100 samples (example)
#define IMU_DIM 6         // ax ay az gx gy gz

float imu_buffer[WINDOW_SIZE][IMU_DIM];
int buffer_index = 0;

void setup() {
  Serial.begin(115200);
  imu_init();
  inflate_init();
}

void loop() {
  float imu_sample[IMU_DIM];

  if (imu_read(imu_sample)) {
    
    // store sample in circular buffer
    for(int i = 0; i < IMU_DIM; i++)
      imu_buffer[buffer_index][i] = imu_sample[i];
    
    buffer_index = (buffer_index + 1) % WINDOW_SIZE;

    // Only evaluate when buffer just wrapped = full window collected
    if (buffer_index == 0) {
      float score = run_model((float*)imu_buffer);

      Serial.print("Fall score: ");
      Serial.println(score);

      if (score > FALL_THRESHOLD) {
        trigger_inflate();
      }
    }
  }
}
