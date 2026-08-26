Parallel Neural Network Inference Engine

A Parallel and Distributed Computing course project that investigates how different CPU parallelization strategies affect the performance of Convolutional Neural Network (CNN) inference.

The project uses Fashion-MNIST for a 10-class image-classification task. A small CNN is trained using PyTorch and CUDA, its learned parameters are exported, and the forward pass is then implemented from scratch in C++. The C++ inference engine is used as the baseline for sequential and OpenMP-based parallel experiments.

Project Objectives

The main objectives are:

Train a lightweight CNN on Fashion-MNIST.

Use CUDA/GPU acceleration during the training phase.

Export the trained model parameters independently of PyTorch inference.

Implement the CNN forward pass in C++.

Establish a correct sequential CPU inference baseline.

Investigate multiple OpenMP parallelization strategies:

Intra-layer parallelism

Batch parallelism

Pipeline parallelism

Measure:

Execution time

Speedup

Parallel efficiency

Throughput

Scalability with different thread counts

Analyze the point at which additional parallelism stops providing proportional performance improvements.

Dataset

The project uses Fashion-MNIST.

Dataset characteristics

Property

Value

Training images

60,000

Test images

10,000

Image size

28 × 28

Channels

1

Classes

10

Pixel range

0–255

The dataset is stored locally and is intentionally not committed to GitHub.

Expected local directory:

faishon-dataset/
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte

Note: the directory is currently named faishon-dataset in this project.

CNN Architecture

The project uses a deliberately small CNN so that the focus remains on inference performance and parallelization rather than model complexity.

Input
1 × 28 × 28
     │
     ▼
Conv2D
1 → 8 filters
3 × 3 kernel
     │
     ▼
ReLU
     │
     ▼
MaxPool
2 × 2
     │
     ▼
Conv2D
8 → 16 filters
3 × 3 kernel
     │
     ▼
ReLU
     │
     ▼
MaxPool
2 × 2
     │
     ▼
Flatten
16 × 5 × 5 = 400
     │
     ▼
Fully Connected
400 → 64
     │
     ▼
ReLU
     │
     ▼
Fully Connected
64 → 10
     │
     ▼
Class prediction

Total trainable parameters:

27,562

Training

Training is performed using:

Python

PyTorch

CUDA

NVIDIA GPU

Adam optimizer

Cross-entropy loss

Fashion-MNIST

The current training environment used an NVIDIA GeForce RTX 4050 Laptop GPU with a CUDA-enabled PyTorch installation.

Current model result

The improved training run achieved:

Metric

Result

Best test accuracy

90.92%

Best test loss

0.2573

Final training accuracy

92.96%

Training time

58.50 s

Parameters

27,562

The best model is:

training/models/fashion_cnn_best.pth

Model files are intentionally excluded from GitHub.

Training Pipeline

Fashion-MNIST
      │
      ▼
CSV Dataset
      │
      ▼
PyTorch Dataset
      │
      ▼
DataLoader
      │
      ▼
CNN
      │
      ▼
CUDA / RTX 4050
      │
      ▼
Training
      │
      ▼
Best model
fashion_cnn_best.pth

Weight Export

The trained PyTorch model is not used directly by the C++ inference engine.

Instead, the trained parameters are exported as raw float32 binary files.

fashion_cnn_best.pth
        │
        ▼
export_weights.py
        │
        ├── conv1_weight.bin
        ├── conv1_bias.bin
        ├── conv2_weight.bin
        ├── conv2_bias.bin
        ├── fc1_weight.bin
        ├── fc1_bias.bin
        ├── fc2_weight.bin
        └── fc2_bias.bin

Expected parameter shapes:

Parameter

Shape

conv1_weight

(8, 1, 3, 3)

conv1_bias

(8)

conv2_weight

(16, 8, 3, 3)

conv2_bias

(16)

fc1_weight

(64, 400)

fc1_bias

(64)

fc2_weight

(10, 64)

fc2_bias

(10)

The binary weights are also excluded from GitHub.

C++ Inference Engine

The C++ implementation will reproduce the CNN forward pass without relying on PyTorch.

The intended sequential execution is:

Input
  │
  ▼
Conv2D
  │
  ▼
ReLU
  │
  ▼
MaxPool
  │
  ▼
Conv2D
  │
  ▼
ReLU
  │
  ▼
MaxPool
  │
  ▼
Flatten
  │
  ▼
Dense
  │
  ▼
ReLU
  │
  ▼
Dense
  │
  ▼
Prediction

The C++ implementation will first be validated against the PyTorch model.

The goal is:

Same input
    │
    ├───────────────┐
    ▼               ▼
PyTorch           C++
    │               │
    ▼               ▼
Prediction       Prediction
    │               │
    └───────┬───────┘
            ▼
         MATCH

Only after correctness is established will parallelization be introduced.

Parallelization Strategies

1. Sequential Baseline

The complete CNN forward pass executes sequentially on the CPU.

This provides the baseline:

T_serial

All subsequent speedup calculations are based on this implementation.

2. Intra-Layer Parallelism

Independent operations inside a neural-network layer are distributed among OpenMP threads.

Examples include:

output convolution channels

output pixels

dense-layer neurons

Conceptually:

             Conv2D
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
    Thread 0 Thread 1 Thread 2 ...
       │        │        │
       └────────┼────────┘
                ▼
            Output

3. Batch Parallelism

Different input images are processed concurrently.

Image 0 ──► Thread 0
Image 1 ──► Thread 1
Image 2 ──► Thread 2
Image 3 ──► Thread 3
...

This strategy is particularly relevant when multiple inference requests are available.

4. Pipeline Parallelism

Different stages of inference are organized as pipeline stages.

Conceptually:

Stage 1       Stage 2       Stage 3       Stage 4

Conv1  ───►   Pool1  ───►  Conv2  ───►  Dense
   │             │            │            │
Image 1       Image 1      Image 1      Image 1
Image 2       Image 2      Image 2      Image 2
Image 3       Image 3      Image 3      Image 3

The pipeline is intended to improve throughput when multiple inputs are processed.

Performance Metrics

The project will measure the following.

Execution Time

T = total inference execution time

Speedup

Speedup(p) = T_serial / T_parallel(p)

where p is the number of OpenMP threads.

Parallel Efficiency

Efficiency(p) = Speedup(p) / p

Throughput

Throughput = Number of processed samples / Total time

Scalability

The implementation will be evaluated using different thread counts, for example:

1
2
4
8
12
16
...

The exact thread counts will depend on the available CPU hardware.

Project Structure

PDC-Project/
│
├── faishon-dataset/
│
├── training/
│   ├── model.py
│   ├── train.py
│   ├── test_model.py
│   ├── export_weights.py
│   │
│   └── models/
│       ├── fashion_cnn_best.pth
│       └── weights/
│
├── inference/
│   ├── main.cpp
│   ├── tensor.h
│   ├── tensor.cpp
│   │
│   ├── layers/
│   │   ├── convolution.h
│   │   ├── convolution.cpp
│   │   ├── activation.h
│   │   ├── activation.cpp
│   │   ├── pooling.h
│   │   ├── pooling.cpp
│   │   ├── dense.h
│   │   ├── dense.cpp
│   │   ├── softmax.h
│   │   └── softmax.cpp
│   │
│   ├── sequential/
│   ├── parallel/
│   └── pipeline/
│
├── benchmarks/
│   ├── benchmark.cpp
│   └── results/
│
├── analysis/
│   ├── analyze.py
│   ├── plots.py
│   └── results.csv
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_layers.cpp
│   └── test_inference.cpp
│
├── docs/
├── CMakeLists.txt
├── requirements.txt
├── project_structure.sh
└── README.md

Software Requirements

Python

Python 3.x

PyTorch

pandas

NumPy

C++

GCC / G++

C++17 or newer

OpenMP

CMake

The current development environment uses:

GCC       13.3.0
OpenMP    4.5
PyTorch   2.5.1+cu121
CUDA      12.1 PyTorch build
GPU       NVIDIA GeForce RTX 4050 Laptop GPU

Building the C++ Project

The C++ build system is based on CMake.

The intended workflow is:

mkdir -p build
cd build
cmake ..
make -j$(nproc)

The exact build targets will be added as the inference engine is implemented.

Running Training

From the project root:

cd training
python3 train.py

To export the trained model parameters:

python3 export_weights.py

Development Methodology

The project is being developed incrementally.

Phase 1 — Machine Learning

Dataset preparation

CNN design

GPU training

Accuracy evaluation

Model checkpointing

Phase 2 — Model Export

Extract PyTorch parameters

Convert parameters to raw float32 binary files

Validate parameter dimensions

Phase 3 — Sequential C++ Inference

Tensor abstraction

Binary weight loading

Convolution

ReLU

Max pooling

Flattening

Dense layers

Prediction

PyTorch/C++ correctness validation

Phase 4 — Parallel Inference

OpenMP intra-layer parallelism

OpenMP batch parallelism

OpenMP pipeline parallelism

Phase 5 — Benchmarking

Thread-count experiments

Execution time measurements

Speedup

Efficiency

Throughput

Scalability analysis

Phase 6 — Analysis

Compare all execution strategies

Identify bottlenecks

Analyze synchronization and memory effects

Study diminishing returns

Relate observations to Amdahl's Law

Current Status

[✓] Fashion-MNIST dataset prepared
[✓] CNN implemented in PyTorch
[✓] CUDA/GPU training working
[✓] 30-epoch improved training run
[✓] 90.92% best test accuracy achieved
[✓] Best model saved
[✓] Model weights exported to binary files
[✓] C++ compiler verified
[✓] OpenMP support verified
[✓] Project structure created
[✓] Git repository initialized
[✓] Initial commits pushed to GitHub

[ ] C++ Tensor implementation
[ ] Binary weight loader
[ ] C++ Conv2D
[ ] C++ ReLU
[ ] C++ MaxPool
[ ] C++ Dense layers
[ ] Sequential inference
[ ] PyTorch vs C++ validation
[ ] OpenMP intra-layer implementation
[ ] OpenMP batch implementation
[ ] OpenMP pipeline implementation
[ ] Benchmarking
[ ] Performance analysis
[ ] Final report

Expected Final Comparison

The final project will compare:

                    CNN Inference
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     Sequential      Intra-layer      Batch
          │           OpenMP          OpenMP
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                      Pipeline
                      OpenMP
                         │
                         ▼
                   Benchmarking
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Time           Speedup       Efficiency
                         │
                         ▼
                    Scalability

The final analysis will determine which strategy provides the best performance under different workloads and thread counts, and where additional parallelism begins to provide diminishing returns.

Git and Large Files

The following files are intentionally excluded from Git:

faishon-dataset/
training/models/*.pth
training/models/**/*.bin

The repository contains the source code and scripts required to reproduce the training/export workflow rather than committing the dataset and generated model artifacts.

License

This project is developed as an academic Parallel and Distributed Computing course project.