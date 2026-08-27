#include "dense.h"
#include <iostream>
#include <omp.h>

Dense::Dense(const std::string& weight_path, const std::string& bias_path, std::vector<size_t> w_shape)
    : weights(w_shape), biases({w_shape[0]}) {
    if (!weights.load(weight_path)) std::cerr << "Failed to load dense weights" << std::endl;
    if (!biases.load(bias_path)) std::cerr << "Failed to load dense biases" << std::endl;
}

Tensor Dense::forward(const Tensor& input, ParallelMode mode, int batch_idx) const {
    size_t batch = input.shape[0];
    size_t in_features = input.shape[1];
    size_t out_features = weights.shape[0];
    
    Tensor output({batch, out_features});
    
    size_t b_start = (batch_idx == -1) ? 0 : batch_idx;
    size_t b_end = (batch_idx == -1) ? batch : batch_idx + 1;
    
    if (mode == ParallelMode::INTRA_LAYER) {
        #pragma omp parallel for
        for (size_t of = 0; of < out_features; ++of) {
            for (size_t b = b_start; b < b_end; ++b) {
                float sum = biases.get(of);
                for (size_t inf = 0; inf < in_features; ++inf) {
                    sum += input.get2d(b, inf) * weights.get2d(of, inf);
                }
                output.get2d(b, of) = sum;
            }
        }
    } else {
        #pragma omp parallel for if(mode == ParallelMode::BATCH && batch_idx == -1)
        for (size_t b = b_start; b < b_end; ++b) {
            for (size_t of = 0; of < out_features; ++of) {
                float sum = biases.get(of);
                for (size_t inf = 0; inf < in_features; ++inf) {
                    sum += input.get2d(b, inf) * weights.get2d(of, inf);
                }
                output.get2d(b, of) = sum;
            }
        }
    }
    return output;
}
