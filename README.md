# Parallel Neural Network Inference Engine

A Parallel and Distributed Computing course project focused on implementing and evaluating different parallelization strategies for Convolutional Neural Network (CNN) inference using C++ and OpenMP.

The project uses **Fashion-MNIST** for image classification. A lightweight CNN is trained using **PyTorch + CUDA**, its learned weights are exported, and the same model will be implemented in C++ for sequential and parallel inference experiments.

---

## Project Overview

```text
                    Fashion-MNIST
                         |
                         v
                  PyTorch + CUDA
                         |
                         v
                    CNN Training
                         |
                         v
                  Trained Model
                         |
                         v
                  Weight Export
                         |
                         v
                C++ Inference Engine
                         |
              +----------+----------+
              |          |          |
              v          v          v
         Sequential   Intra-Layer  Batch
          Baseline     OpenMP      OpenMP
              |          |          |
              +----------+----------+
                         |
                         v
                  Pipeline OpenMP
                         |
                         v
                   Benchmarking
                         |
                         v
               Performance Analysis
```

The main goal is to investigate how different CPU parallelization strategies affect CNN inference performance.

---

## Objectives

- Train a lightweight CNN on Fashion-MNIST.
- Use CUDA/GPU acceleration during training.
- Export the trained model parameters as binary files.
- Implement the CNN forward pass from scratch in C++.
- Establish a sequential CPU inference baseline.
- Implement OpenMP-based parallel inference.
- Compare different parallelization strategies.
- Measure execution time, speedup, efficiency, and throughput.
- Study scalability with different numbers of CPU threads.
- Analyze bottlenecks and diminishing returns from parallelism.

---

## Dataset

The project uses the **Fashion-MNIST** dataset.

| Property | Value |
|---|---:|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 x 28 |
| Channels | 1 |
| Classes | 10 |
| Pixel values | 0-255 |

The dataset is stored locally and is not committed to GitHub.

```text
faishon-dataset/
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

> Note: the directory is currently named `faishon-dataset` in this project.

---

# CNN Architecture

The project uses a lightweight CNN so that the primary focus remains on inference performance and parallelization.

```text
Input
1 x 28 x 28
      |
      v
Conv2D
1 -> 8 filters
3 x 3 kernel
      |
      v
ReLU
      |
      v
MaxPool
2 x 2
      |
      v
Conv2D
8 -> 16 filters
3 x 3 kernel
      |
      v
ReLU
      |
      v
MaxPool
2 x 2
      |
      v
Flatten
16 x 5 x 5 = 400
      |
      v
Fully Connected
400 -> 64
      |
      v
ReLU
      |
      v
Fully Connected
64 -> 10
      |
      v
Prediction
```

### Model Parameters

```text
27,562 trainable parameters
```

---

# Training

The CNN is trained using:

- Python
- PyTorch
- CUDA
- NVIDIA GeForce RTX 4050 Laptop GPU
- Fashion-MNIST
- Cross-Entropy Loss
- Adam optimizer

## Current Training Results

The current best model achieved:

| Metric | Result |
|---|---:|
| Best Test Accuracy | **90.92%** |
| Best Test Loss | **0.2573** |
| Final Training Accuracy | **92.96%** |
| Training Time | **58.50 seconds** |
| Parameters | **27,562** |

The best trained model is stored locally as:

```text
training/models/fashion_cnn_best.pth
```

Model files are excluded from GitHub.

---

# Training Pipeline

```text
Fashion-MNIST
      |
      v
CSV Dataset
      |
      v
PyTorch Dataset
      |
      v
DataLoader
      |
      v
CNN Model
      |
      v
CUDA / RTX 4050
      |
      v
Training
      |
      v
Best Model
```

### Train the model

```bash
cd training
python3 train.py
```

---

# Model Weight Export

The C++ inference engine does not depend on PyTorch during inference.

The trained model parameters are exported from PyTorch into raw binary files.

```text
fashion_cnn_best.pth
        |
        v
export_weights.py
        |
        +-- conv1_weight.bin
        +-- conv1_bias.bin
        +-- conv2_weight.bin
        +-- conv2_bias.bin
        +-- fc1_weight.bin
        +-- fc1_bias.bin
        +-- fc2_weight.bin
        +-- fc2_bias.bin
```

### Parameter Shapes

| Parameter | Shape |
|---|---|
| `conv1_weight` | `(8, 1, 3, 3)` |
| `conv1_bias` | `(8)` |
| `conv2_weight` | `(16, 8, 3, 3)` |
| `conv2_bias` | `(16)` |
| `fc1_weight` | `(64, 400)` |
| `fc1_bias` | `(64)` |
| `fc2_weight` | `(10, 64)` |
| `fc2_bias` | `(10)` |

### Export weights

```bash
cd training
python3 export_weights.py
```

The generated binary weight files are kept locally and ignored by Git.

---

# C++ Inference Engine

The next major stage is implementing the CNN forward pass from scratch in C++.

The intended inference pipeline is:

```text
Input
  |
  v
Conv2D
  |
  v
ReLU
  |
  v
MaxPool
  |
  v
Conv2D
  |
  v
ReLU
  |
  v
MaxPool
  |
  v
Flatten
  |
  v
Dense
  |
  v
ReLU
  |
  v
Dense
  |
  v
Prediction
```

The C++ implementation will first be validated against the PyTorch model.

```text
              Same Input
                  |
          +-------+-------+
          |               |
          v               v
       PyTorch           C++
          |               |
          v               v
     Prediction       Prediction
          |               |
          +-------+-------+
                  |
                  v
               Compare
                  |
                  v
              Must Match
```

Only after correctness is established will OpenMP parallelization be introduced.

---

# Parallelization Strategies

## 1. Sequential Baseline

The complete CNN forward pass will first execute sequentially on the CPU.

This provides the baseline:

```text
T_serial
```

All parallel speedup calculations will be based on this implementation.

---

## 2. Intra-Layer Parallelism

Independent operations within a neural-network layer will be distributed among OpenMP threads.

Potential parallelization targets include:

- Output convolution channels
- Output pixels
- Convolution operations
- Dense-layer neurons

Conceptually:

```text
                    Conv2D
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Thread 0    Thread 1    Thread 2
          |           |           |
          +-----------+-----------+
                      |
                      v
                    Output
```

---

## 3. Batch Parallelism

Different input images can be processed concurrently.

```text
Image 0 ----------------> Thread 0
Image 1 ----------------> Thread 1
Image 2 ----------------> Thread 2
Image 3 ----------------> Thread 3
...
```

This strategy is particularly useful when multiple inference requests are available.

---

## 4. Pipeline Parallelism

Different CNN stages can be organized into pipeline stages.

```text
Input
  |
  v
[ Conv1 ] -> [ Pool1 ] -> [ Conv2 ] -> [ Pool2 ] -> [ Dense ]
    |           |           |           |           |
    v           v           v           v           v
 Stage 1     Stage 2     Stage 3     Stage 4     Stage 5
```

The objective is to determine whether pipeline execution can improve inference throughput for multiple inputs.

---

# Performance Evaluation

The implementations will be evaluated using the following metrics.

## Execution Time

```text
T = Total inference execution time
```

## Speedup

```text
Speedup(p) = T_serial / T_parallel(p)
```

where `p` is the number of OpenMP threads.

## Parallel Efficiency

```text
Efficiency(p) = Speedup(p) / p
```

## Throughput

```text
Throughput = Number of Samples / Total Execution Time
```

## Scalability

The implementation will be evaluated using different numbers of OpenMP threads, for example:

```text
1
2
4
8
12
16
...
```

The exact thread counts will depend on the available CPU hardware.

---

# Project Structure

```text
PDC-Project/
|
+-- faishon-dataset/
|
+-- training/
|   +-- model.py
|   +-- train.py
|   +-- test_model.py
|   +-- export_weights.py
|   +-- models/
|
+-- inference/
|   +-- main.cpp
|   +-- tensor.cpp
|   +-- tensor.h
|   |
|   +-- layers/
|   |   +-- convolution.cpp
|   |   +-- convolution.h
|   |   +-- activation.cpp
|   |   +-- activation.h
|   |   +-- pooling.cpp
|   |   +-- pooling.h
|   |   +-- dense.cpp
|   |   +-- dense.h
|   |   +-- softmax.cpp
|   |   +-- softmax.h
|   |
|   +-- sequential/
|   +-- parallel/
|   +-- pipeline/
|
+-- benchmarks/
|   +-- benchmark.cpp
|   +-- results/
|
+-- analysis/
|   +-- analyze.py
|   +-- plots.py
|
+-- tests/
|   +-- test_tensor.cpp
|   +-- test_layers.cpp
|   +-- test_inference.cpp
|
+-- docs/
+-- CMakeLists.txt
+-- requirements.txt
+-- project_structure.sh
+-- .gitignore
+-- README.md
```

---

# Development Roadmap

## Phase 1 - Model Training

- [x] Fashion-MNIST dataset prepared
- [x] CNN implemented using PyTorch
- [x] CUDA training configured
- [x] Model trained
- [x] 90.92% best test accuracy achieved

## Phase 2 - Model Export

- [x] Trained model saved
- [x] Weight export implemented
- [x] Binary weight files generated

## Phase 3 - C++ Inference

- [ ] Tensor implementation
- [ ] Binary weight loader
- [ ] Conv2D implementation
- [ ] ReLU implementation
- [ ] MaxPool implementation
- [ ] Dense layer implementation
- [ ] Sequential inference
- [ ] PyTorch vs C++ validation

## Phase 4 - Parallel Inference

- [ ] OpenMP intra-layer parallelism
- [ ] OpenMP batch parallelism
- [ ] OpenMP pipeline parallelism

## Phase 5 - Benchmarking

- [ ] Sequential benchmark
- [ ] Thread-scaling experiments
- [ ] Speedup measurements
- [ ] Parallel efficiency measurements
- [ ] Throughput measurements
- [ ] Scalability analysis

## Phase 6 - Final Analysis

- [ ] Compare sequential and parallel approaches
- [ ] Identify performance bottlenecks
- [ ] Analyze synchronization overhead
- [ ] Analyze memory-access overhead
- [ ] Study diminishing returns
- [ ] Prepare final report

---

# Technologies

| Technology | Purpose |
|---|---|
| Python | Model training and analysis |
| PyTorch | CNN implementation |
| CUDA | GPU-accelerated training |
| C++ | Custom inference engine |
| OpenMP | CPU parallelization |
| CMake | C++ build system |
| Fashion-MNIST | Dataset |
| Git / GitHub | Version control |

---

# Development Environment

```text
Operating System : Ubuntu 24.04
Python            : 3.11.14
PyTorch           : 2.5.1+cu121
CUDA Build        : 12.1
GPU               : NVIDIA GeForce RTX 4050 Laptop GPU
GCC               : 13.3.0
OpenMP            : 4.5
```

---

# Building the C++ Project

The C++ project uses CMake.

Once the inference implementation is ready:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

The available build targets will be expanded as the C++ inference engine is implemented.

---

# Git and Large Files

The following generated or large files are intentionally excluded from version control:

```text
faishon-dataset/
training/models/*.pth
training/models/**/*.bin
```

The repository contains the source code, training scripts, inference implementation, tests, and benchmarking tools.

---

# Expected Final Experiment

The final experiment will compare:

```text
                    CNN Inference
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     Sequential      Intra-Layer      Batch
       Baseline        OpenMP        OpenMP
          |              |              |
          +--------------+--------------+
                         |
                         v
                    Pipeline OpenMP
                         |
                         v
                    Benchmarking
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      Execution        Speedup       Efficiency
        Time
                         |
                         v
                    Scalability
```

The final objective is to determine which parallelization strategy provides the best performance for CNN inference and how performance changes as the number of CPU threads increases.

---

# Current Status

The machine-learning preparation stage is complete.

```text
Fashion-MNIST
     |
     v
PyTorch CNN
     |
     v
CUDA Training
     |
     v
90.92% Test Accuracy
     |
     v
Weight Export
     |
     v
C++ Inference Engine  <-- CURRENT STAGE
     |
     +-- Sequential
     |
     +-- OpenMP Intra-Layer
     |
     +-- OpenMP Batch
     |
     +-- OpenMP Pipeline
```

### Current milestone

The next development milestone is:

**Implement the C++ Tensor class, binary weight loader, and sequential CNN inference engine.**

---

## Academic Project

This project is developed as part of a **Parallel and Distributed Computing** course project.
