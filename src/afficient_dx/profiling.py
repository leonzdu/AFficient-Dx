import importlib
import importlib.metadata
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from afficient_dx.inference import logits_from_model

def layer_report(model, length, in_channels=12):
    model = model.cpu().eval()
    rows, hooks = ([], [])
    seen = set()
    for name, module in model.named_modules():
        direct = list(module.parameters(recurse=False))
        if list(module.children()) and (not direct):
            continue
        unique = [p for p in direct if id(p) not in seen]
        seen.update((id(p) for p in unique))
        row = {'layer': name or '<root>', 'type': type(module).__name__, 'parameters': sum((p.numel() for p in unique)), 'trainable_parameters': sum((p.numel() for p in unique if p.requires_grad)), 'parameter_bytes': sum((p.numel() * p.element_size() for p in unique)), 'buffer_bytes': sum((b.numel() * b.element_size() for b in module.buffers(recurse=False))), 'input_shape': None, 'output_shape': None, 'conv_linear_macs': 0}
        rows.append(row)

        def hook(m, inputs, out, row=row):
            if inputs and torch.is_tensor(inputs[0]):
                row['input_shape'] = list(inputs[0].shape)
            if torch.is_tensor(out):
                row['output_shape'] = list(out.shape)
            if isinstance(m, nn.Conv1d):
                row['conv_linear_macs'] += out.numel() * m.in_channels // m.groups * m.kernel_size[0]
            elif isinstance(m, nn.Linear):
                row['conv_linear_macs'] += out.numel() * m.in_features
        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            logits_from_model(model, torch.zeros(1, in_channels, length))
    finally:
        for handle in hooks:
            handle.remove()
    return rows

def count_macs(model, length, in_channels=12):
    return int(sum((row['conv_linear_macs'] for row in layer_report(model, length, in_channels))))

def runtime_metadata():
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'wfdb', 'matplotlib']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = platform.processor()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    return {'python': sys.version, 'platform': platform.platform(), 'cpu': cpu, 'machine': platform.machine(), 'logical_cpus': os.cpu_count(), 'packages': versions, 'torch_cuda': torch.version.cuda, 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch_threads': torch.get_num_threads(), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

@torch.inference_mode()
def benchmark_cpu(model, sample, warmup, repeats, threads):
    if repeats == 0:
        return {'status': 'disabled'}
    previous_threads = torch.get_num_threads()
    model = model.cpu().eval()
    sample = torch.from_numpy(np.ascontiguousarray(sample[:1])).cpu()
    try:
        torch.set_num_threads(threads)
        for _ in range(warmup):
            model(sample)
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample)
            elapsed.append((time.perf_counter_ns() - start) / 1000000.0)
    finally:
        torch.set_num_threads(previous_threads)
    return {'status': 'measured', 'device': 'cpu', 'batch_size': 1, 'threads': threads, 'input_shape': list(sample.shape), 'warmup': warmup, 'repeats': repeats, 'mean_ms': float(np.mean(elapsed)), 'std_ms': float(np.std(elapsed)), 'median_ms': float(np.median(elapsed)), 'p95_ms': float(np.percentile(elapsed, 95)), 'scope': 'eager model forward including Python overhead; excludes preprocessing/I/O', 'clock': 'perf_counter_ns', 'hardware': runtime_metadata()}
