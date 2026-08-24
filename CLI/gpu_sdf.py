#!/usr/bin/env python3
# GPU-based SDF via progressive dilation. Replaces CPU scipy EDT.
import torch
import torch.nn.functional as F


def gpu_sdf(binary_gpu, max_sweeps=80):
    solid = (binary_gpu > 0.5).float()
    fluid = 1.0 - solid
    INF = float(max(binary_gpu.shape) * 3)
    dist_out = torch.full_like(binary_gpu, INF)
    dist_out[solid > 0.5] = 0.0
    dist_in = torch.full_like(binary_gpu, INF)
    dist_in[fluid > 0.5] = 0.0
    cur_solid = solid.unsqueeze(0).unsqueeze(0)
    cur_fluid = fluid.unsqueeze(0).unsqueeze(0)
    done_out = False
    done_in = False
    for step in range(1, max_sweeps + 1):
        if not done_out:
            new_s = F.max_pool3d(cur_solid, 3, stride=1, padding=1)
            newly_o = (new_s.squeeze(0).squeeze(0) > 0.5) & (dist_out >= INF - 0.5)
            if newly_o.any():
                dist_out[newly_o] = float(step)
            else:
                done_out = True
            cur_solid = new_s
        if not done_in:
            new_f = F.max_pool3d(cur_fluid, 3, stride=1, padding=1)
            newly_i = (new_f.squeeze(0).squeeze(0) > 0.5) & (dist_in >= INF - 0.5)
            if newly_i.any():
                dist_in[newly_i] = float(step)
            else:
                done_in = True
            cur_fluid = new_f
        if done_out and done_in:
            break
    return dist_out - dist_in
