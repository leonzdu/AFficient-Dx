import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import DataLoader, TensorDataset

def logits_from_model(model, x):
    result = model(x)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or result.shape[0] != x.shape[0]:
        raise ValueError('Model must return one binary logit per ECG (not probabilities)')
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite model logits')
    return result

def make_loader(x, y, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator if shuffle else None)

@torch.inference_mode()
def predict_logits(model, loader, device):
    model.eval()
    ys, zs = ([], [])
    for x, y in loader:
        ys.append(y.numpy())
        zs.append(logits_from_model(model, x.to(device)).cpu().numpy())
    return (np.concatenate(ys), np.concatenate(zs).astype(np.float64))

def predict(model, loader, device):
    y, z = predict_logits(model, loader, device)
    return (y, expit(z))
