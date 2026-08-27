#include "pooling.h"
#include <algorithm>
#include <limits>
#include <omp.h>

MaxPool2D::MaxPool2D(int kernel_size, int stride) : kernel_size(kernel_size), stride(stride) {}

Tensor MaxPool2D::forward(const Tensor& input, ParallelMode mode, int batch_idx) const {
    size_t batch = input.shape[0];
    size_t channels = input.shape[1];
    size_t in_h = input.shape[2];
    size_t in_w = input.shape[3];
    
    size_t out_h = (in_h - kernel_size) / stride + 1;
    size_t out_w = (in_w - kernel_size) / stride + 1;
    
    Tensor output({batch, channels, out_h, out_w});
    
    size_t b_start = (batch_idx == -1) ? 0 : batch_idx;
    size_t b_end = (batch_idx == -1) ? batch : batch_idx + 1;
    
    if (mode == ParallelMode::INTRA_LAYER) {
        #pragma omp parallel for
        for (size_t c = 0; c < channels; ++c) {
            for (size_t b = b_start; b < b_end; ++b) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float max_val = std::numeric_limits<float>::lowest();
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                float val = input.get4d(b, c, ih, iw);
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                        output.get4d(b, c, oh, ow) = max_val;
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for if(mode == ParallelMode::BATCH && batch_idx == -1)
        for (size_t b = b_start; b < b_end; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float max_val = std::numeric_limits<float>::lowest();
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                float val = input.get4d(b, c, ih, iw);
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                        output.get4d(b, c, oh, ow) = max_val;
                    }
                }
            }
        }
    }
    return output;
}
