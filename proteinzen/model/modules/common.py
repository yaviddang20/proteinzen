import torch

torch._dynamo.config.cache_size_limit = 16

try:
    cuda_major, cuda_minor = torch.cuda.get_device_capability(device=None)
    if cuda_major >= 8:
        compile_supported = True
    else:
        compile_supported = False
except RuntimeError:
    compile_supported = False
print(f"using torch.compile: {compile_supported}")

try:
    cuda_major, cuda_minor = torch.cuda.get_device_capability(device=None)
    if cuda_major >= 8:
        cuet_supported = True
    else:
        cuet_supported = False
except RuntimeError:
    cuet_supported = False
print(f"using cuet kernels: {cuet_supported}")
