import importlib
import importlib.metadata
import json
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, c1, c2, stride=1, residual=True, batchnorm=True):
        super().__init__()
        norm = nn.BatchNorm1d if batchnorm else lambda _: nn.Identity()
        self.main = nn.Sequential(nn.Conv1d(c1, c2, 7, stride, 3, bias=False), norm(c2), nn.ReLU(), nn.Conv1d(c2, c2, 5, 1, 2, bias=False), norm(c2))
        self.residual = residual
        self.skip = (nn.Identity() if c1 == c2 and stride == 1 else nn.Sequential(nn.Conv1d(c1, c2, 1, stride, bias=False), norm(c2))) if residual else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        return self.relu(out + self.skip(x) if self.residual else out)

class ECGResNet(nn.Module):

    def __init__(self, base=32, dropout=0.15, in_channels=12, channels=None, residual=True, batchnorm=True):
        super().__init__()
        channels = tuple(channels or (base, 2 * base, 4 * base))
        if not channels or any((c < 1 for c in channels)):
            raise ValueError('Every channel width must be a positive integer')
        layers = [nn.Conv1d(in_channels, channels[0], 15, 2, 7, bias=False), nn.BatchNorm1d(channels[0]) if batchnorm else nn.Identity(), nn.ReLU()]
        previous = channels[0]
        for i, width in enumerate(channels):
            layers.append(Block(previous, width, 1 if i == 0 else 2, residual, batchnorm))
            previous = width
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        return self.fc(self.dropout(self.features(x).squeeze(-1))).squeeze(-1)

def make_model(args, length):
    if args.model_factory:
        module, function = args.model_factory.rsplit(':', 1)
        factory = getattr(importlib.import_module(module), function)
        model = factory(in_channels=len(args.leads), length=length, frequency=args.frequency, **json.loads(args.model_kwargs))
        if not isinstance(model, nn.Module):
            raise TypeError('External model factory must return torch.nn.Module')
        return model
    return ECGResNet(args.base, args.dropout, len(args.leads), args.channels, not args.no_residual, not args.no_batchnorm)
