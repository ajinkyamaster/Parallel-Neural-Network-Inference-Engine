import torch
from model import FashionCNN


model = FashionCNN()

x = torch.randn(4, 1, 28, 28)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)
print(model)