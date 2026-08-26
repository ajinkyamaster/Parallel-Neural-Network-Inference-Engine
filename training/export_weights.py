import os
import torch

from model import FashionCNN


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "models/fashion_cnn_best.pth"

OUTPUT_DIR = "models/weights"


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("Loading trained model")
print("=" * 60)

model = FashionCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Extract weights
# ============================================================

weights = {

    "conv1_weight":
        model.conv1.weight.detach().numpy(),

    "conv1_bias":
        model.conv1.bias.detach().numpy(),

    "conv2_weight":
        model.conv2.weight.detach().numpy(),

    "conv2_bias":
        model.conv2.bias.detach().numpy(),

    "fc1_weight":
        model.fc1.weight.detach().numpy(),

    "fc1_bias":
        model.fc1.bias.detach().numpy(),

    "fc2_weight":
        model.fc2.weight.detach().numpy(),

    "fc2_bias":
        model.fc2.bias.detach().numpy(),
}


# ============================================================
# Print shapes
# ============================================================

print("\nWeight shapes:")

for name, value in weights.items():

    print(
        f"{name:15s} : "
        f"{value.shape}"
    )


# ============================================================
# Save weights
# ============================================================

for name, value in weights.items():

    path = os.path.join(
        OUTPUT_DIR,
        name + ".bin"
    )

    value.astype("float32").tofile(path)

    print(
        f"Saved: {path}"
    )


# ============================================================
# Done
# ============================================================

print("\n" + "=" * 60)
print("Weight export completed")
print("=" * 60)