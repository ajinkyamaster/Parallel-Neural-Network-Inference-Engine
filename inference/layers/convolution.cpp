#include "convolution.h"
#include <iostream>
#include <omp.h>

Conv2D::Conv2D(const std::string& weight_path, const std::string& bias_path, std::vector<size_t> w_shape, int stride, int padding) 
    : weights(w_shape), biases({w_shape[0]}), stride(stride), padding(padding) {
    if (!weights.load(weight_path)) std::cerr << "Failed to load conv weights" << std::endl;
    if (!biases.load(bias_path)) std::cerr << "Failed to load conv biases" << std::endl;
}

Tensor Conv2D::forward(const Tensor& input, ParallelMode mode, int batch_idx) const {
    size_t batch = input.shape[0];
    size_t in_channels = input.shape[1];
    size_t in_h = input.shape[2];
    size_t in_w = input.shape[3];
    
    size_t out_channels = weights.shape[0];
    size_t kernel_h = weights.shape[2];
    size_t kernel_w = weights.shape[3];
    
    size_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
    
    Tensor output({batch, out_channels, out_h, out_w});
    
    size_t b_start = (batch_idx == -1) ? 0 : batch_idx;
    size_t b_end = (batch_idx == -1) ? batch : batch_idx + 1;
    
    auto compute_element = [&](size_t b, size_t oc, size_t oh, size_t ow) {
        float sum = biases.get(oc);
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                        sum += input.get4d(b, ic, ih, iw) * weights.get4d(oc, ic, kh, kw);
                    }
                }
            }
        }
        output.get4d(b, oc, oh, ow) = sum;
    };
    
    if (mode == ParallelMode::INTRA_LAYER) {
        #pragma omp parallel for
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t b = b_start; b < b_end; ++b) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        compute_element(b, oc, oh, ow);
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for if(mode == ParallelMode::BATCH && batch_idx == -1)
        for (size_t b = b_start; b < b_end; ++b) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        compute_element(b, oc, oh, ow);
                    }
                }
            }
        }
    }
    return output;
}
