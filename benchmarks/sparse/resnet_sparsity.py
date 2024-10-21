import torch
import torchvision.models as models
import time
from torch.sparse import to_sparse_semi_structured
import torch.nn as nn

def generate_random_input(batch_size=1):
    # Generate a random input tensor
    return torch.randn(batch_size, 3, 224, 224)

def run_resnet_inference(model, input_tensor):
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    return output

def measure_inference_time(model, input_tensor, num_iterations):
    # Measure inference time
    start_time = time.time()
    for _ in range(num_iterations):
        run_resnet_inference(model, input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_iterations

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True).to(device).half()
    model.eval()

    # Generate random input
    batch_size = 32
    input_tensor = generate_random_input(batch_size).to(device).half()

    # Warm-up run
    run_resnet_inference(model, input_tensor)

    # Number of iterations for averaging
    num_iterations = 100

    # Measure baseline FP16 inference time
    avg_inference_time_fp16 = measure_inference_time(model, input_tensor, num_iterations)
    print(f"FP16 Average inference time: {avg_inference_time_fp16:.4f} seconds")
    print(f"FP16 Throughput: {batch_size / avg_inference_time_fp16:.2f} images/second")

    # Apply 2:4 sparsity
    print("Applying 2:4 sparsity...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Create 2:4 sparsity mask
            mask = torch.Tensor([0, 0, 1, 1]).tile((module.weight.shape[0], module.weight.shape[1])).cuda().bool()
            module.weight = nn.Parameter(mask * module.weight)

    # Measure sparse inference time (before acceleration)
    avg_inference_time_sparse_before = measure_inference_time(model, input_tensor, num_iterations)
    print(f"Sparse (before acceleration) Average inference time: {avg_inference_time_sparse_before:.4f} seconds")
    print(f"Sparse (before acceleration) Throughput: {batch_size / avg_inference_time_sparse_before:.2f} images/second")

    # Accelerate via SparseSemiStructuredTensor
    print("Accelerating with SparseSemiStructuredTensor...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))

    # Measure sparse inference time (after acceleration)
    avg_inference_time_sparse_after = measure_inference_time(model, input_tensor, num_iterations)
    print(f"Sparse (after acceleration) Average inference time: {avg_inference_time_sparse_after:.4f} seconds")
    print(f"Sparse (after acceleration) Throughput: {batch_size / avg_inference_time_sparse_after:.2f} images/second")

    # Calculate and print speedup
    speedup = avg_inference_time_fp16 / avg_inference_time_sparse_after
    print(f"\nSpeedup: {speedup:.3f}x")

    # Measure memory usage if CUDA is available
    if torch.cuda.is_available():
        memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        print(f"Max GPU memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    main()
