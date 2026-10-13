#include "mpi_pipeline.h"
#include <iostream>
#include <algorithm>

MPIPipelineInference::MPIPipelineInference(const std::string& weights_dir, MPI_Comm comm)
    : rank(0), size(1), comm(comm), weights_dir(weights_dir),
      conv1(weights_dir + "/conv1_weight.bin", weights_dir + "/conv1_bias.bin", {8, 1, 3, 3}, 1, 0),
      pool1(2, 2),
      conv2(weights_dir + "/conv2_weight.bin", weights_dir + "/conv2_bias.bin", {16, 8, 3, 3}, 1, 0),
      pool2(2, 2),
      fc1(weights_dir + "/fc1_weight.bin", weights_dir + "/fc1_bias.bin", {64, 400}),
      fc2(weights_dir + "/fc2_weight.bin", weights_dir + "/fc2_bias.bin", {10, 64}) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
}

Tensor MPIPipelineInference::forward(const Tensor& global_input, MPIPipelineTiming& timing) {
    int total_batch = 0;
    if (rank == 0) {
        total_batch = static_cast<int>(global_input.shape.empty() ? 0 : global_input.shape[0]);
    }
    
    MPI_Bcast(&total_batch, 1, MPI_INT, 0, comm);
    
    if (total_batch == 0 || size < 4) {
        if (rank == 0 && size < 4) {
            std::cerr << "[MPI Pipeline Error] Pipeline mode requires at least 4 MPI processes (current: " << size << ")\n";
        }
        timing = {0.0, 0.0, 0.0};
        return Tensor();
    }
    
    MPI_Barrier(comm);
    double t_start = MPI_Wtime();
    
    Tensor global_output;
    if (rank == 0) {
        global_output = Tensor({static_cast<size_t>(total_batch), 10});
        
        // Stage 1: Process images and forward activation tensors to Rank 1
        for (int b = 0; b < total_batch; ++b) {
            Tensor single_img({1, 1, 28, 28});
            for (size_t i = 0; i < 28 * 28; ++i) {
                single_img.get(i) = global_input.get4d(b, 0, i / 28, i % 28);
            }
            
            Tensor s1 = conv1.forward(single_img);
            s1 = relu1.forward(s1);
            s1 = pool1.forward(s1); // shape: [1, 8, 13, 13] -> 1352 floats
            
            MPI_Send(s1.data.data(), 1352, MPI_FLOAT, 1, b, comm);
        }
        
        // Wait for final results from Rank 3 (Stage 4)
        MPI_Recv(global_output.data.data(), total_batch * 10, MPI_FLOAT, 3, 99999, comm, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        // Stage 2: Receive from Rank 0, compute Conv2+ReLU2+Pool2, send to Rank 2
        for (int b = 0; b < total_batch; ++b) {
            Tensor s1({1, 8, 13, 13});
            MPI_Recv(s1.data.data(), 1352, MPI_FLOAT, 0, b, comm, MPI_STATUS_IGNORE);
            
            Tensor s2 = conv2.forward(s1);
            s2 = relu2.forward(s2);
            s2 = pool2.forward(s2); // shape: [1, 16, 5, 5] -> 400 floats
            
            MPI_Send(s2.data.data(), 400, MPI_FLOAT, 2, b, comm);
        }
    }
    else if (rank == 2) {
        // Stage 3: Receive from Rank 1, compute Flatten+FC1+ReLU3, send to Rank 3
        for (int b = 0; b < total_batch; ++b) {
            Tensor s2_flat({1, 400});
            MPI_Recv(s2_flat.data.data(), 400, MPI_FLOAT, 1, b, comm, MPI_STATUS_IGNORE);
            
            Tensor s3 = fc1.forward(s2_flat);
            s3 = relu3.forward(s3); // shape: [1, 64] -> 64 floats
            
            MPI_Send(s3.data.data(), 64, MPI_FLOAT, 3, b, comm);
        }
    }
    else if (rank == 3) {
        // Stage 4: Receive from Rank 2, compute FC2+Softmax, collect and send back to Rank 0
        std::vector<float> final_results(total_batch * 10);
        for (int b = 0; b < total_batch; ++b) {
            Tensor s3({1, 64});
            MPI_Recv(s3.data.data(), 64, MPI_FLOAT, 2, b, comm, MPI_STATUS_IGNORE);
            
            Tensor s4 = fc2.forward(s3);
            s4 = softmax.forward(s4); // shape: [1, 10] -> 10 floats
            
            for (size_t c = 0; c < 10; ++c) {
                final_results[b * 10 + c] = s4.get2d(0, c);
            }
        }
        
        MPI_Send(final_results.data(), total_batch * 10, MPI_FLOAT, 0, 99999, comm);
    }
    
    MPI_Barrier(comm);
    double t_end = MPI_Wtime();
    
    if (rank == 0) {
        timing.total_time = t_end - t_start;
        timing.latency_per_image = (timing.total_time * 1000.0) / total_batch;
        timing.throughput = total_batch / timing.total_time;
    }
    
    return global_output;
}
