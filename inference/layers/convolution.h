#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../tensor.h"
#include "../parallel_mode.h"

class Conv2D {
public:
    Tensor weights;
    Tensor biases;
    
    int stride;
    int padding;
    
    Conv2D(const std::string& weight_path, const std::string& bias_path, std::vector<size_t> w_shape, int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input, ParallelMode mode = ParallelMode::SEQUENTIAL, int batch_idx = -1) const;
};

#endif
