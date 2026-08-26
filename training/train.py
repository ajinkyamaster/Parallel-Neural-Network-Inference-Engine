import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import FashionCNN


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "../faishon-dataset/fashion-mnist_train.csv"
TEST_CSV = "../faishon-dataset/fashion-mnist_test.csv"

BATCH_SIZE = 128
EPOCHS = 30

LEARNING_RATE = 0.001

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fashion_cnn_best.pth"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Fashion-MNIST CNN Training")
print("=" * 60)

print("PyTorch version :", torch.__version__)
print("Device          :", device)

if torch.cuda.is_available():

    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA build      :",
        torch.version.cuda
    )


# ============================================================
# Dataset
# ============================================================

class FashionMNISTCSV(Dataset):

    def __init__(self, csv_file):

        print(f"\nLoading: {csv_file}")

        data = pd.read_csv(csv_file)

        # First column = label
        self.labels = torch.tensor(
            data.iloc[:, 0].values,
            dtype=torch.long
        )

        # Remaining 784 columns = pixels
        self.images = torch.tensor(
            data.iloc[:, 1:].values,
            dtype=torch.float32
        )

        # Normalize pixels
        self.images /= 255.0

        # 784 → 1 × 28 × 28
        self.images = self.images.reshape(
            -1,
            1,
            28,
            28
        )

        print(
            "Images shape :",
            self.images.shape
        )

        print(
            "Labels shape :",
            self.labels.shape
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.images[index],
            self.labels[index]
        )


# ============================================================
# Load datasets
# ============================================================

train_dataset = FashionMNISTCSV(
    TRAIN_CSV
)

test_dataset = FashionMNISTCSV(
    TEST_CSV
)

print(
    "\nTraining samples:",
    len(train_dataset)
)

print(
    "Testing samples :",
    len(test_dataset)
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# Model
# ============================================================

model = FashionCNN().to(device)

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print(model)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nTotal parameters:",
    num_parameters
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Best model tracking
# ============================================================

best_accuracy = 0.0
best_loss = float("inf")


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

total_training_start = time.perf_counter()


for epoch in range(EPOCHS):

    epoch_start = time.perf_counter()

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    # --------------------------------------------------------
    # Training batches
    # --------------------------------------------------------

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # Forward
        outputs = model(images)


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # Statistics
        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # Validation / Test
    # --------------------------------------------------------

    model.eval()

    test_loss = 0.0

    test_correct = 0
    test_total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            test_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            test_total += labels.size(0)

            test_correct += (
                predictions == labels
            ).sum().item()


    test_loss /= test_total

    test_accuracy = (
        100.0
        * test_correct
        / test_total
    )


    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler.step(test_loss)


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:2d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.2f}s"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy
        best_loss = test_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  >>> New best model saved! "
            f"Accuracy: {best_accuracy:.2f}%"
        )


# ============================================================
# Final results
# ============================================================

total_training_time = (
    time.perf_counter()
    - total_training_start
)


print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Training time : "
    f"{total_training_time:.2f} seconds"
)

print(
    f"Best test loss: "
    f"{best_loss:.4f}"
)

print(
    f"Best accuracy : "
    f"{best_accuracy:.2f}%"
)

print(
    "\nBest model saved to:"
)

print(
    os.path.abspath(MODEL_PATH)
)

print("=" * 60)