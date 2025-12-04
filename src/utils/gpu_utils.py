import torch

def autodetect_num_envs(target_mem_util_ratio: float = 0.8, base_mem_mb_per_env: float = 5.0, min_envs: int = 1, max_envs: int = 1024) -> int:
    """
    Dynamically determines the number of parallel environments based on available GPU memory.

    Args:
        target_mem_util_ratio (float): The desired ratio of total GPU memory to utilize (e.g., 0.8 for 80%).
        base_mem_mb_per_env (float): Estimated base memory (in MB) required per environment.
                                     This includes mesh generation and feature extraction.
        min_envs (int): Minimum number of environments to return.
        max_envs (int): Maximum number of environments to return.

    Returns:
        int: The recommended number of environments.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, defaulting to min_envs for CPU usage.")
        return min_envs

    try:
        # Get GPU device properties
        device = torch.device('cuda')
        device_properties = torch.cuda.get_device_properties(device)
        total_gpu_memory_bytes = device_properties.total_memory
        total_gpu_memory_mb = total_gpu_memory_bytes / (1024 * 1024)

        # Estimate available memory for environments
        # We assume some memory is used by the agent itself and other system processes
        # For simplicity, we directly use the target_mem_util_ratio on total memory.
        # A more sophisticated approach would query torch.cuda.memory_stats to get actual free memory.
        usable_gpu_memory_mb = total_gpu_memory_mb * target_mem_util_ratio

        # Calculate potential number of environments
        if base_mem_mb_per_env <= 0:
            estimated_num_envs = max_envs # Avoid division by zero or negative
        else:
            estimated_num_envs = int(usable_gpu_memory_mb / base_mem_mb_per_env)

        # Clamp between min and max
        num_envs = max(min_envs, min(max_envs, estimated_num_envs))
        
        print(f"GPU: {device_properties.name}, Total Memory: {total_gpu_memory_mb:.2f} MB")
        print(f"Estimated usable memory for environments: {usable_gpu_memory_mb:.2f} MB (Target utilization: {target_mem_util_ratio*100:.0f}%)")
        print(f"Recommended parallel environments: {num_envs}")
        return num_envs

    except Exception as e:
        print(f"Error autodetecting GPU memory: {e}. Defaulting to min_envs.")
        return min_envs
