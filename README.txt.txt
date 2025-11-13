Workspace/
│
├── Firmware/                     # Everything that runs on the microcontroller
│   ├── src/
│   │   ├── main.cpp              # main loop + system init
│   │   ├── fsm.h                 # state machine definitions
│   │   ├── fsm.cpp               # fall logic, deployment logic
│   │
│   │   ├── imu.h                 # IMU interface (read sensors)
│   │   ├── imu.cpp               
│   │
│   │   ├── processing.h          # ANY processing: filters, windows, features
│   │   ├── processing.cpp
│   │
│   │   ├── model_interface.h     
│   │   ├── model_interface.cpp   # wraps whatever model Bill decides to use
│   │
│   │   ├── inflate.h             # inflator (CO₂ trigger) or could be peripherals
│   │   ├── inflate.cpp
│   │
│   │   ├── ble_notify.h          # optional: BLE notifications
│   │   ├── ble_notify.cpp
│   │
│   │   ├── logger.h              # optional logging (serial, SD card)
│   │   ├── logger.cpp
│   │
│   │   └── config.h              # system constants, thresholds
│   │
│   └── model/
│       ├── model_params.h        # Any parameters Bill gives you (weights, thresholds)
│       ├── model_code.cpp        # Model logic (NN, thresholding, tree, etc.)
│       └── metadata.h            # input size, version, config info
│
│
└── ML_Model/                      # Bill will do this.
   