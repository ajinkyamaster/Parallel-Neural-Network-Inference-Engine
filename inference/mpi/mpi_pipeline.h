#ifndef MPI_PIPELINE_H
#define MPI_PIPELINE_H

#include <mpi.h>
#include <string>
#include <vector>
#include "../tensor.h"
#include "../layers/convolution.h"
#include "../layers/pooling.h"
#include "../layers/activation.h"
#include "../layers/dense.h"

struct MPIPipelineTiming {
    double total_time;
    double latency_per_image;
    double throughput;
};

class MPIPipelineInference {
public:
    MPIPipelineInference(const std::string& weights_dir, MPI_Comm comm = MPI_COMM_WORLD);
    
    // Executes 4-stage pipeline inference across ranks 0, 1, 2, 3
    // Rank 0 feeds input images and receives final results
    Tensor forward(const Tensor& global_input, MPIPipelineTiming& timing);

    int get_rank() const { return rank; }
    int get_size() const { return size; }

private:
    int rank;
    int size;
    MPI_Comm comm;
    std::string weights_dir;
    
    // Stage 1 layers (Rank 0)
    Conv2D conv1;
    ReLU relu1;
    MaxPool2D pool1;
    
    // Stage 2 layers (Rank 1)
    Conv2D conv2;
    ReLU relu2;
    MaxPool2D pool2;
    
    // Stage 3 layers (Rank 2)
    Dense fc1;
    ReLU relu3;
    
    // Stage 4 layers (Rank 3)
    Dense fc2;
    Softmax softmax;
};

#endif // MPI_PIPELINE_H
