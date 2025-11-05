Workspace/
│
├── Firmware/
│   ├── src/
│   │   ├── main.ino
│   │   ├── imu.h
│   │   ├── imu.cpp
│   │   ├── inflate.h
│   │   ├── inflate.cpp
│   │   ├── model_inference.h
│   │   └── model_inference.cpp
│   │
│   └── model/
│       ├── model_weights.h      
│       └── tinycnn_forward.cpp  
│
└── ML_Model/                    
    ├── train_model.py
    ├── export_to_c_array.py
    ├── data/
    │   └── (dataset files go here)
    └── saved_models/
        └── best_model.pt