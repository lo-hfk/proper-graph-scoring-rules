"""check whether old inference and new one are equivalent for one torchvision and one timm model"""

import numpy as np

# pick the two models that were run twice
for name in ["shufflenet_v2_x1_0", "efficientnet_b2"]:
    old = np.load(f"examples/imagenet-1k/model_validation_outputs/probs_{name}.npy")
    new = np.load(f"examples/imagenet/data/probs/{name}.npy")
    print(f"{name}: max diff = {np.max(np.abs(old - new)):.2e}, identical = {np.array_equal(old, new)}")
