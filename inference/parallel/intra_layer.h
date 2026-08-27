#ifndef INTRA_LAYER_H
#define INTRA_LAYER_H

#include "../tensor.h"
#include "../layers/convolution.h"
#include "../layers/dense.h"
#include "../layers/pooling.h"
#include "../layers/activation.h"
#include "../parallel_mode.h"
#include <string>

class IntraLayerInference {
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
    
    IntraLayerInference(const std::string& weights_dir);
    
    Tensor forward(const Tensor& input) const;
};

#endif
