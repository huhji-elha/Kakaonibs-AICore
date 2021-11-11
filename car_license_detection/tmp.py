import detect_image
import torch

with torch.no_grad():
    res = detect_image.detect(source="")

print(res)
