#ifndef DENSE_H
#define DENSE_H

#include "../tensor.h"
#include "../parallel_mode.h"

class Dense {
public:
    Tensor weights;
    Tensor biases;
    
    Dense(const std::string& weight_path, const std::string& bias_path, std::vector<size_t> w_shape);
    
    Tensor forward(const Tensor& input, ParallelMode mode = ParallelMode::SEQUENTIAL, int batch_idx = -1) const;
};

#endif
