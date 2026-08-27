#include "activation.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

Tensor ReLU::forward(const Tensor& input, ParallelMode mode, int batch_idx) const {
    Tensor output = input;
    
    if (batch_idx != -1) {
        size_t b = batch_idx;
        size_t elements_per_batch = input.size() / input.shape[0];
        size_t start = b * elements_per_batch;
        size_t end = start + elements_per_batch;
        
        #pragma omp parallel for if(mode == ParallelMode::INTRA_LAYER)
        for (size_t i = start; i < end; ++i) {
            if (output.get(i) < 0.0f) {
                output.get(i) = 0.0f;
            }
        }
        return output;
    }
    
    // Process full batch
    if (mode == ParallelMode::BATCH) {
        size_t batch = input.shape[0];
        size_t elements_per_batch = input.size() / batch;
        
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            size_t start = b * elements_per_batch;
            size_t end = start + elements_per_batch;
            for (size_t i = start; i < end; ++i) {
                if (output.get(i) < 0.0f) {
                    output.get(i) = 0.0f;
                }
            }
        }
    } else {
        // Includes INTRA_LAYER over the full flattened tensor
        // This is perfectly valid as one big chunked loop
        #pragma omp parallel for if(mode == ParallelMode::INTRA_LAYER)
        for (size_t i = 0; i < output.size(); ++i) {
            if (output.get(i) < 0.0f) {
                output.get(i) = 0.0f;
            }
        }
    }
    
    return output;
}

Tensor Softmax::forward(const Tensor& input, ParallelMode mode, int batch_idx) const {
    Tensor output = input;
    size_t batch = input.shape[0];
    size_t classes = input.shape[1];
    
    size_t b_start = (batch_idx == -1) ? 0 : batch_idx;
    size_t b_end = (batch_idx == -1) ? batch : batch_idx + 1;
    
    // Softmax is applied per image.
    // In intra-layer, it's hard to parallelize effectively without reduction, 
    // but we can just parallelize over the batch loop if we treat it as "batch of neurons" for softmax?
    // Actually, softmax for a single image has very few elements (10). 
    // Parallelizing over classes is too small. 
    // So for INTRA_LAYER, parallelizing the outer b loop here makes the most sense as well.
    // The instructions say "Enter parallel region once per layer call" which this does.
    #pragma omp parallel for if(mode == ParallelMode::BATCH && batch_idx == -1 || mode == ParallelMode::INTRA_LAYER)
    for (size_t b = b_start; b < b_end; ++b) {
        float max_val = input.get2d(b, 0);
        for (size_t c = 1; c < classes; ++c) {
            if (input.get2d(b, c) > max_val) {
                max_val = input.get2d(b, c);
            }
        }
        
        float sum_exp = 0.0f;
        for (size_t c = 0; c < classes; ++c) {
            float e = std::exp(input.get2d(b, c) - max_val);
            output.get2d(b, c) = e;
            sum_exp += e;
        }
        
        for (size_t c = 0; c < classes; ++c) {
            output.get2d(b, c) /= sum_exp;
        }
    }
    return output;
}
