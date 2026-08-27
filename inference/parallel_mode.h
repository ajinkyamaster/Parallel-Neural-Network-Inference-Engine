#ifndef PARALLEL_MODE_H
#define PARALLEL_MODE_H

enum class ParallelMode {
    SEQUENTIAL,
    INTRA_LAYER,
    BATCH,
    PIPELINE
};

#endif
