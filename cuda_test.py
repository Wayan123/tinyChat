import torch

# Periksa apakah CUDA (GPU) tersedia
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU (CUDA) tersedia.")
else:
    device = torch.device("cpu")
    print("GPU (CUDA) tidak tersedia, menggunakan CPU.")

# Buat tensor di GPU (CUDA) jika tersedia
if device.type == "cuda":
    x = torch.rand(3, 3).to(device)
    print("Tensor di GPU:")
    print(x)
else:
    x = torch.rand(3, 3)
    print("Tensor di CPU:")
    print(x)
