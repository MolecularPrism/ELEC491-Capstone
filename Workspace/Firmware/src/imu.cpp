#include "imu.h"
#include <Arduino_LSM6DSOX.h> // Built-in IMU

bool imu_init() {
  return LSM6DSOX.begin();
}

bool imu_read(float *buf) {
  float ax, ay, az, gx, gy, gz;

  if (!LSM6DSOX.accelerationAvailable() || !LSM6DSOX.gyroscopeAvailable())
    return false;

  LSM6DSOX.readAcceleration(ax, ay, az);
  LSM6DSOX.readGyroscope(gx, gy, gz);

  buf[0] = ax;
  buf[1] = ay;
  buf[2] = az;
  buf[3] = gx;
  buf[4] = gy;
  buf[5] = gz;

  return true;
}
