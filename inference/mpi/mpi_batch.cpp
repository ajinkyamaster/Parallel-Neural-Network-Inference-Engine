#include "mpi_batch.h"
#include <iostream>
#include <algorithm>

MPIBatchInference::MPIBatchInference(const std::string& weights_dir, MPI_Comm comm)
    : comm(comm), local_engine(weights_dir) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
}

Tensor MPIBatchInference::forward(const Tensor& global_input, MPITimingInfo& timing) {
    int total_batch = 0;
    if (rank == 0) {
        total_batch = static_cast<int>(global_input.shape.empty() ? 0 : global_input.shape[0]);
    }
    
    // Broadcast total batch size from rank 0 to all ranks
    MPI_Bcast(&total_batch, 1, MPI_INT, 0, comm);
    
    if (total_batch == 0) {
        timing = {0.0, 0.0, 0.0, 0.0, 0.0};
        return Tensor();
    }
    
    // Determine partition counts and displacements
    int rem = total_batch % size;
    int local_count = total_batch / size + (rank < rem ? 1 : 0);
    
    std::vector<int> sendcounts(size, 0);
    std::vector<int> sdispls(size, 0);
    std::vector<int> recvcounts(size, 0);
    std::vector<int> rdispls(size, 0);
    
    if (rank == 0) {
        int cur_offset = 0;
        for (int i = 0; i < size; ++i) {
            int cnt = total_batch / size + (i < rem ? 1 : 0);
            sendcounts[i] = cnt * 784;
            sdispls[i] = cur_offset * 784;
            recvcounts[i] = cnt * 10;
            rdispls[i] = cur_offset * 10;
            cur_offset += cnt;
        }
    }
    
    // Allocate local input buffer
    Tensor local_input({static_cast<size_t>(local_count), 1, 28, 28});
    
    // Synchronize before timing to ensure accurate measurement
    MPI_Barrier(comm);
    double t_start = MPI_Wtime();
    
    // 1. Scatter input images (Communication phase 1)
    double t_scatter_0 = MPI_Wtime();
    const float* sendbuf = (rank == 0 && !global_input.data.empty()) ? global_input.data.data() : nullptr;
    MPI_Scatterv(sendbuf, sendcounts.data(), sdispls.data(), MPI_FLOAT,
                 local_input.data.data(), local_count * 784, MPI_FLOAT,
                 0, comm);
    double t_scatter_1 = MPI_Wtime();
    double local_scatter_time = t_scatter_1 - t_scatter_0;
    
    // 2. Local Forward Inference (Computation phase)
    double t_comp_0 = MPI_Wtime();
    Tensor local_output;
    if (local_count > 0) {
        local_output = local_engine.forward(local_input);
    } else {
        local_output = Tensor({0, 10});
    }
    double t_comp_1 = MPI_Wtime();
    double local_compute_time = t_comp_1 - t_comp_0;
    
    // 3. Gather predictions to Rank 0 (Communication phase 2)
    Tensor global_output;
    if (rank == 0) {
        global_output = Tensor({static_cast<size_t>(total_batch), 10});
    }
    
    double t_gather_0 = MPI_Wtime();
    float* recvbuf = (rank == 0) ? global_output.data.data() : nullptr;
    MPI_Gatherv(local_output.data.data(), local_count * 10, MPI_FLOAT,
                recvbuf, recvcounts.data(), rdispls.data(), MPI_FLOAT,
                0, comm);
    double t_gather_1 = MPI_Wtime();
    double local_gather_time = t_gather_1 - t_gather_0;
    
    MPI_Barrier(comm);
    double t_end = MPI_Wtime();
    double local_total_time = t_end - t_start;
    
    // Collect max times across all ranks for critical path metrics
    double max_scatter, max_compute, max_gather, max_total;
    MPI_Reduce(&local_scatter_time, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_gather_time,  &max_gather,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_total_time,   &max_total,   1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    if (rank == 0) {
        timing.total_time = max_total;
        timing.scatter_time = max_scatter;
        timing.compute_time = max_compute;
        timing.gather_time = max_gather;
        timing.comm_time = max_scatter + max_gather;
    }
    
    return global_output;
}
