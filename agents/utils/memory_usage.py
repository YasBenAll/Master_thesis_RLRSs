import psutil
import torch
import GPUtil

def log_memory_usage(make_file=False, file_path=None, step=None, tag=None):
    if make_file:
        with open(file_path, 'w') as f:
            f.write("Memory Usage\n")
    # Get CPU memory usage
    cpu_memory = psutil.virtual_memory()
    total_cpu_memory = cpu_memory.total / (1024 ** 3)  # Convert to GB
    available_cpu_memory = cpu_memory.available / (1024 ** 3)  # Convert to GB
    used_cpu_memory = cpu_memory.used / (1024 ** 3)  # Convert to GB
    memory_percent = cpu_memory.percent  # Percentage of memory used

    # print(f"Total CPU memory: {total_cpu_memory:.2f} GB")
    # print(f"Available CPU memory: {available_cpu_memory:.2f} GB")
    # print(f"Used CPU memory: {used_cpu_memory:.2f} GB")
    # print(f"CPU memory usage: {memory_percent}%")
    with open(file_path, 'a') as f:
        f.write(f"\nStep: {step}\n{tag}\n")
        f.write(f"Total CPU memory: {total_cpu_memory:.2f} GB\n")
        f.write(f"Available CPU memory: {available_cpu_memory:.2f} GB\n")
        f.write(f"Used CPU memory: {used_cpu_memory:.2f} GB\n")
        f.write(f"CPU memory usage: {memory_percent}%\n")

    # Check if CUDA is available and get GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Convert to GB
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
            gpu_memory_free = gpu_memory - gpu_memory_allocated
            total_gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB

            # print(f"\nGPU {i} ({torch.cuda.get_device_name(i)}) memory usage:")
            # print(f"  Total GPU memory: {total_gpu_memory:.2f} GB")
            # print(f"  Allocated GPU memory: {gpu_memory_allocated:.2f} GB")
            # print(f"  Reserved GPU memory: {gpu_memory:.2f} GB")
            # print(f"  Free GPU memory: {gpu_memory_free:.2f} GB")
    else:
        print("CUDA not available. No GPU memory to report.")

    # Optional: Detailed GPU usage using GPUtil (if installed)
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            with open(file_path, 'a') as f:
                f.write(f"\nGPUtil: GPU {gpu.id} ({gpu.name})\n")
                f.write(f"  GPU Load: {gpu.load * 100:.1f}%\n")
                f.write(f"  GPU Memory Used: {gpu.memoryUsed:.2f} GB\n")
                f.write(f"  GPU Memory Free: {gpu.memoryFree:.2f} GB\n")
                f.write(f"  GPU Memory Total: {gpu.memoryTotal:.2f} GB\n")
                f.write(f"  GPU Temperature: {gpu.temperature} °\n")

            # print(f"\nGPUtil: GPU {gpu.id} ({gpu.name})")
            # print(f"  GPU Load: {gpu.load * 100:.1f}%")
            # print(f"  GPU Memory Used: {gpu.memoryUsed:.2f} GB")
            # print(f"  GPU Memory Free: {gpu.memoryFree:.2f} GB")
            # print(f"  GPU Memory Total: {gpu.memoryTotal:.2f} GB")
            # print(f"  GPU Temperature: {gpu.temperature} °C")
    except Exception as e:
        print(f"GPUtil error: {e}")

if __name__ == "__main__":
    log_memory_usage()
