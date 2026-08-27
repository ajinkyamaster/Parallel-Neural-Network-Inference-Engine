#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../tensor.h"
#include "../parallel_mode.h"

class ReLU {
public:
    Tensor forward(const Tensor& input, ParallelMode mode = ParallelMode::SEQUENTIAL, int batch_idx = -1) const;
};

class Softmax {
public:
    Tensor forward(const Tensor& input, ParallelMode mode = ParallelMode::SEQUENTIAL, int batch_idx = -1) const;
};

#endif
