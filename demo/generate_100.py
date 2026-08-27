import numpy as np
import os

os.makedirs('images_100', exist_ok=True)
data = np.fromfile('../training/models/data/test_images.bin', dtype=np.float32)
data = data.reshape(-1, 28*28)
for i in range(100):
    data[i].tofile(f'images_100/image_{i}.bin')
print("Demo images created in demo/images_100/")
