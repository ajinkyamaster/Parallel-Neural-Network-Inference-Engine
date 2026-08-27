#ifndef POOLING_H
#define POOLING_H

#include "../tensor.h"
#include "../parallel_mode.h"

class MaxPool2D {
public:
    int kernel_size;
    int stride;
    
    MaxPool2D(int kernel_size, int stride);
    
    Tensor forward(const Tensor& input, ParallelMode mode = ParallelMode::SEQUENTIAL, int batch_idx = -1) const;
};

#endif
