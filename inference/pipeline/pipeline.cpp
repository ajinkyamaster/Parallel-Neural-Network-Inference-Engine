#include "pipeline.h"
#include <omp.h>
#include <vector>

PipelineInference::PipelineInference(const std::string& weights_dir)
    : conv1(weights_dir + "/conv1_weight.bin", weights_dir + "/conv1_bias.bin", {8, 1, 3, 3}, 1, 0),
      pool1(2, 2),
      conv2(weights_dir + "/conv2_weight.bin", weights_dir + "/conv2_bias.bin", {16, 8, 3, 3}, 1, 0),
      pool2(2, 2),
      fc1(weights_dir + "/fc1_weight.bin", weights_dir + "/fc1_bias.bin", {64, 400}),
      fc2(weights_dir + "/fc2_weight.bin", weights_dir + "/fc2_bias.bin", {10, 64}) {
}

Tensor PipelineInference::forward(const Tensor& input, ParallelMode mode) const {
    size_t batch = input.shape[0];
    Tensor output({batch, 10});
    
    // We will simulate 4 pipeline stages.
    // To enforce a true pipeline structure where stage S of image I depends on stage S of image I-1
    // (to prevent all tasks of stage 1 running concurrently and acting like batch parallelism),
    // we add dependencies across images for the same stage.
    
    std::vector<int> stage1_vec(batch, 0);
    std::vector<int> stage2_vec(batch, 0);
    std::vector<int> stage3_vec(batch, 0);
    std::vector<int> stage4_vec(batch, 0);
    
    int* s1 = stage1_vec.data();
    int* s2 = stage2_vec.data();
    int* s3 = stage3_vec.data();
    int* s4 = stage4_vec.data();
    
    int requested_threads = omp_get_max_threads();
    int threads_per_stage = std::max(1, requested_threads / 4);
    
    #pragma omp parallel
    #pragma omp single
    {
        for (size_t b = 0; b < batch; ++b) {
            Tensor* t1 = new Tensor();
            Tensor* t2 = new Tensor();
            Tensor* t3 = new Tensor();
            
            // Stage 1: Conv1 + ReLU1 + Pool1
            #pragma omp task depend(in: s1[b > 0 ? b - 1 : 0]) depend(out: s1[b])
            {
                omp_set_num_threads(threads_per_stage);
                Tensor single_img({1, 1, 28, 28});
                for (size_t i = 0; i < 28 * 28; ++i) {
                    single_img.get(i) = input.get4d(b, 0, i / 28, i % 28);
                }
                *t1 = conv1.forward(single_img, mode);
                *t1 = relu1.forward(*t1, mode);
                *t1 = pool1.forward(*t1, mode);
            }
            
            // Stage 2: Conv2 + ReLU2 + Pool2
            #pragma omp task depend(in: s1[b], s2[b > 0 ? b - 1 : 0]) depend(out: s2[b])
            {
                omp_set_num_threads(threads_per_stage);
                *t2 = conv2.forward(*t1, mode);
                *t2 = relu2.forward(*t2, mode);
                *t2 = pool2.forward(*t2, mode);
                delete t1;
            }
            
            // Stage 3: Flatten + FC1 + ReLU3
            #pragma omp task depend(in: s2[b], s3[b > 0 ? b - 1 : 0]) depend(out: s3[b])
            {
                omp_set_num_threads(threads_per_stage);
                Tensor flattened({1, 16 * 5 * 5});
                flattened.data = std::move(t2->data);
                delete t2;
                
                *t3 = fc1.forward(flattened, mode);
                *t3 = relu3.forward(*t3, mode);
            }
            
            // Stage 4: FC2 + Softmax
            #pragma omp task depend(in: s3[b], s4[b > 0 ? b - 1 : 0]) depend(out: s4[b])
            {
                omp_set_num_threads(threads_per_stage);
                Tensor final_out = fc2.forward(*t3, mode);
                final_out = softmax.forward(final_out, mode);
                delete t3;
                
                for (size_t c = 0; c < 10; ++c) {
                    output.get2d(b, c) = final_out.get2d(0, c);
                }
            }
        }
    }
    
    return output;
}
