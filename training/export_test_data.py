import os
import torch
from torchvision import datasets, transforms
import numpy as np
from model import FashionCNN

# ============================================================
# Configuration
# ============================================================

DATA_DIR = "../faishon-dataset"
OUTPUT_DIR = "models/data"
MODEL_PATH = "models/fashion_cnn_best.pth"

print("=" * 60)
print("Exporting Test Data and PyTorch Predictions")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Load dataset
# ============================================================

print("Loading Fashion-MNIST test dataset...")
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

# We will export the first 10,000 images to match the standard test set
num_samples = len(test_dataset)
images = torch.zeros((num_samples, 1, 28, 28), dtype=torch.float32)
labels = torch.zeros(num_samples, dtype=torch.long)

for i in range(num_samples):
    img, label = test_dataset[i]
    images[i] = img
    labels[i] = label

print(f"Images shape : {images.shape}")
print(f"Labels shape : {labels.shape}")

# ============================================================
# Run through PyTorch Model
# ============================================================

print("Loading trained model...")
device = torch.device("cpu")
model = FashionCNN().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"WARNING: Model {MODEL_PATH} not found. Using randomly initialized weights.")
model.eval()

print("Generating PyTorch predictions...")
batch_size = 1000
predictions = torch.zeros(num_samples, dtype=torch.int32)

with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        batch_imgs = images[i:i+batch_size]
        outputs = model(batch_imgs)
        preds = outputs.argmax(dim=1).to(torch.int32)
        predictions[i:i+batch_size] = preds

# Compute accuracy to show baseline
correct = (predictions == labels.to(torch.int32)).sum().item()
print(f"PyTorch Test Accuracy: {correct / num_samples * 100.0:.2f}%")

# ============================================================
# Save to binary
# ============================================================

images_np = images.numpy()
labels_np = labels.numpy().astype(np.int32)
predictions_np = predictions.numpy()

images_path = os.path.join(OUTPUT_DIR, "test_images.bin")
labels_path = os.path.join(OUTPUT_DIR, "test_labels.bin")
preds_path = os.path.join(OUTPUT_DIR, "test_preds.bin")

images_np.tofile(images_path)
labels_np.tofile(labels_path)
predictions_np.tofile(preds_path)

print(f"Saved: {images_path}")
print(f"Saved: {labels_path}")
print(f"Saved: {preds_path}")

print("\n" + "=" * 60)
print("Test data export completed")
print("=" * 60)
