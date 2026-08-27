#ifndef PIPELINE_H
#define PIPELINE_H

#include "../tensor.h"
#include "../layers/convolution.h"
#include "../layers/dense.h"
#include "../layers/pooling.h"
#include "../layers/activation.h"
#include "../parallel_mode.h"
#include <string>

class PipelineInference {
public:
    Conv2D conv1;
    ReLU relu1;
    MaxPool2D pool1;
    
    Conv2D conv2;
    ReLU relu2;
    MaxPool2D pool2;
    
    Dense fc1;
    ReLU relu3;
    
    Dense fc2;
    Softmax softmax;
    
    PipelineInference(const std::string& weights_dir);
    
    Tensor forward(const Tensor& input, ParallelMode mode = ParallelMode::SEQUENTIAL) const;
};

#endif
