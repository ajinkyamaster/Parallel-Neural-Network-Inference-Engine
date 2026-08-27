#include "batch.h"

BatchInference::BatchInference(const std::string& weights_dir)
    : conv1(weights_dir + "/conv1_weight.bin", weights_dir + "/conv1_bias.bin", {8, 1, 3, 3}, 1, 0),
      pool1(2, 2),
      conv2(weights_dir + "/conv2_weight.bin", weights_dir + "/conv2_bias.bin", {16, 8, 3, 3}, 1, 0),
      pool2(2, 2),
      fc1(weights_dir + "/fc1_weight.bin", weights_dir + "/fc1_bias.bin", {64, 400}),
      fc2(weights_dir + "/fc2_weight.bin", weights_dir + "/fc2_bias.bin", {10, 64}) {
}

Tensor BatchInference::forward(const Tensor& input) const {
    ParallelMode mode = ParallelMode::BATCH;
    
    Tensor x = conv1.forward(input, mode);
    x = relu1.forward(x, mode);
    x = pool1.forward(x, mode);
    
    x = conv2.forward(x, mode);
    x = relu2.forward(x, mode);
    x = pool2.forward(x, mode);
    
    Tensor flattened({x.shape[0], 16 * 5 * 5});
    flattened.data = std::move(x.data);
    
    x = fc1.forward(flattened, mode);
    x = relu3.forward(x, mode);
    
    x = fc2.forward(x, mode);
    x = softmax.forward(x, mode);
    
    return x;
}
