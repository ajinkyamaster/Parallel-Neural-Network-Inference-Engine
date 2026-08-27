#include "sequential.h"

SequentialInference::SequentialInference(const std::string& weights_dir)
    : conv1(weights_dir + "/conv1_weight.bin", weights_dir + "/conv1_bias.bin", {8, 1, 3, 3}, 1, 0),
      pool1(2, 2),
      conv2(weights_dir + "/conv2_weight.bin", weights_dir + "/conv2_bias.bin", {16, 8, 3, 3}, 1, 0),
      pool2(2, 2),
      fc1(weights_dir + "/fc1_weight.bin", weights_dir + "/fc1_bias.bin", {64, 400}),
      fc2(weights_dir + "/fc2_weight.bin", weights_dir + "/fc2_bias.bin", {10, 64}) {
}

Tensor SequentialInference::forward(const Tensor& input) const {
    Tensor x = conv1.forward(input);
    x = relu1.forward(x);
    x = pool1.forward(x);
    
    x = conv2.forward(x);
    x = relu2.forward(x);
    x = pool2.forward(x);
    
    // Flatten: reshape from (batch, 16, 5, 5) to (batch, 400)
    Tensor flattened({x.shape[0], 16 * 5 * 5});
    flattened.data = x.data; // copy data
    
    x = fc1.forward(flattened);
    x = relu3.forward(x);
    
    x = fc2.forward(x);
    x = softmax.forward(x);
    
    return x;
}
