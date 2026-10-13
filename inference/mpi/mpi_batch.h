#ifndef MPI_BATCH_H
#define MPI_BATCH_H

#include <mpi.h>
#include <string>
#include <vector>
#include "../tensor.h"
#include "../sequential/sequential.h"

struct MPITimingInfo {
    double total_time;
    double scatter_time;
    double compute_time;
    double gather_time;
    double comm_time;
};

class MPIBatchInference {
public:
    MPIBatchInference(const std::string& weights_dir, MPI_Comm comm = MPI_COMM_WORLD);
    
    // Executes distributed batch inference across MPI ranks
    // Rank 0 supplies global_input (or only Rank 0 needs full global_input loaded)
    // Returns global_output on Rank 0 (and empty tensor on worker ranks)
    Tensor forward(const Tensor& global_input, MPITimingInfo& timing);

    int get_rank() const { return rank; }
    int get_size() const { return size; }

private:
    int rank;
    int size;
    MPI_Comm comm;
    SequentialInference local_engine;
};

#endif // MPI_BATCH_H
