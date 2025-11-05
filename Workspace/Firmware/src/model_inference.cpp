#include "model_inference.h"
#include "model_weights.h"  // supplied by teammate

const float FALL_THRESHOLD = 0.75; // placeholder

float run_model(float *input) {
    // call tinyML inference (auto-generated)
    return tinycnn_forward(input);
}
