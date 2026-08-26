#!/usr/bin/env bash

set -e

# ============================================================
# Parallel CNN Inference Engine - Project Structure
# ============================================================
# Run this script from the PDC-Project root:
#     bash setup_project.sh
#
# It creates the project structure without deleting or
# overwriting your existing files.
# ============================================================

PROJECT_ROOT="$(pwd)"

echo "Creating project structure in:"
echo "$PROJECT_ROOT"
echo

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

mkdir -p \
    training/models/weights \
    inference/layers \
    inference/sequential \
    inference/parallel \
    inference/pipeline \
    benchmarks/results \
    analysis \
    models \
    docs \
    tests

# ------------------------------------------------------------
# Training files
# ------------------------------------------------------------

touch \
    training/model.py \
    training/train.py \
    training/test_model.py \
    training/export_weights.py

# ------------------------------------------------------------
# C++ inference engine
# ------------------------------------------------------------

touch \
    inference/main.cpp \
    inference/tensor.h \
    inference/tensor.cpp \
    inference/layers/convolution.h \
    inference/layers/convolution.cpp \
    inference/layers/activation.h \
    inference/layers/activation.cpp \
    inference/layers/pooling.h \
    inference/layers/pooling.cpp \
    inference/layers/dense.h \
    inference/layers/dense.cpp \
    inference/layers/softmax.h \
    inference/layers/softmax.cpp \
    inference/sequential/sequential.h \
    inference/sequential/sequential.cpp \
    inference/parallel/intra_layer.h \
    inference/parallel/intra_layer.cpp \
    inference/parallel/batch.h \
    inference/parallel/batch.cpp \
    inference/pipeline/pipeline.h \
    inference/pipeline/pipeline.cpp

# ------------------------------------------------------------
# Benchmarking
# ------------------------------------------------------------

touch \
    benchmarks/benchmark.cpp \
    benchmarks/README.md

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------

touch \
    analysis/plots.py \
    analysis/analyze.py \
    analysis/results.csv

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

touch \
    tests/test_tensor.cpp \
    tests/test_layers.cpp \
    tests/test_inference.cpp

# ------------------------------------------------------------
# Root project files
# ------------------------------------------------------------

touch \
    CMakeLists.txt \
    README.md \
    requirements.txt \
    .gitignore

# ------------------------------------------------------------
# Placeholder files for directories that should remain in Git
# ------------------------------------------------------------

touch \
    training/models/.gitkeep \
    training/models/weights/.gitkeep \
    benchmarks/results/.gitkeep \
    models/.gitkeep

# ------------------------------------------------------------
# Remove accidental placeholder if real weights already exist
# ------------------------------------------------------------

if [ "$(find training/models/weights -maxdepth 1 -type f ! -name '.gitkeep' | wc -l)" -gt 0 ]; then
    rm -f training/models/weights/.gitkeep
fi

echo "Project structure created successfully."
echo
echo "Current structure:"
echo

find . \
    -not -path './.git/*' \
    -not -path './faishon-dataset/*' \
    -not -path './training/__pycache__/*' \
    -print | sort

echo
echo "Dataset directory and existing training/model files were preserved."
echo "No existing files were deleted."