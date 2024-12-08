import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is using CPU.")
