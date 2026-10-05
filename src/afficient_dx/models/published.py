import math
import torch
from torch import nn
from torch.nn import functional as F
from afficient_dx.models.residual import ECGResNet
MODEL_NAMES = ('afficient_419', 'smaller_238', 'basso_ph_multiscopic', 'busia_transformer')
DISPLAY_NAMES = {'afficient_419': 'AFficient-Dx (419)', 'smaller_238': 'AFficient-Dx, reduced channels (238)', 'basso_ph_multiscopic': 'Basso PH-Multi-Scopic n=4 (adapted)', 'busia_transformer': 'Busia tiny transformer (adapted)'}

class PHConv1d(nn.Module):

    def __init__(self, n, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        if in_channels % n or out_channels % n:
            raise ValueError('PH convolution dimensions must be divisible by n')
        self.A = nn.Parameter(torch.empty(n, n, n))
        self.F = nn.Parameter(torch.empty(n, out_channels // n, in_channels // n, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.F)
        nn.init.uniform_(self.bias, -1 / math.sqrt(in_channels), 1 / math.sqrt(in_channels))
        self.out_channels, self.in_channels = (out_channels, in_channels)
        self.kernel_size, self.dilation = (kernel_size, dilation)

    def effective_weight(self):
        return torch.einsum('aij,aopk->iojpk', self.A, self.F).reshape(self.out_channels, self.in_channels, self.kernel_size)

    def forward(self, x):
        return F.conv1d(x, self.effective_weight(), padding=self.dilation * (self.kernel_size - 1) // 2, dilation=self.dilation)

class PHLinear(nn.Module):

    def __init__(self, n, in_features, out_features):
        super().__init__()
        if in_features % n or out_features % n:
            raise ValueError('PH linear dimensions must be divisible by n')
        self.A = nn.Parameter(torch.empty(n, n, n))
        self.S = nn.Parameter(torch.empty(n, out_features // n, in_features // n))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.S)
        nn.init.uniform_(self.bias, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))
        self.out_features, self.in_features = (out_features, in_features)

    def effective_weight(self):
        return torch.einsum('aij,aop->iojp', self.A, self.S).reshape(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

class BassoBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel, dilations, dropout):
        layers = []
        for dilation in dilations:
            layers.extend([PHConv1d(4, in_channels, out_channels, kernel, dilation), nn.BatchNorm1d(out_channels), nn.ReLU()])
            in_channels = out_channels
        layers.extend([nn.BatchNorm1d(out_channels), nn.MaxPool1d(2, 2)])
        if dropout:
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)

class BassoPHMultiScopic(nn.Module):

    def __init__(self, in_channels=12, n_classes=1):
        super().__init__()
        scopes = (((1,), (1, 1), (1, 1, 1)), ((2,), (2, 4), (8, 8, 8)), ((4,), (4, 8), (16, 32, 64)))
        kernels = ((3, 3, 3), (5, 5, 3), (9, 7, 5))
        self.branches = nn.ModuleList()
        for branch in range(3):
            previous = in_channels
            blocks = []
            for width, kernel, dilation, dropout in zip((16, 32, 64), kernels[branch], scopes[branch], (0, 0.2, 0)):
                blocks.append(BassoBlock(previous, width, kernel, dilation, dropout))
                previous = width
            self.branches.append(nn.Sequential(*blocks))
        self.se = nn.Sequential(PHLinear(4, 192, 24), nn.ReLU(), PHLinear(1, 24, 192), nn.Sigmoid())
        self.classifier = nn.Sequential(PHLinear(4, 192, 256), nn.Mish(), nn.Dropout(0.2), PHLinear(4, 256, 64), nn.Mish(), nn.Dropout(0.2), PHLinear(1, 64, n_classes))

    def forward(self, x, aux=None):
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        features = features * self.se(features.mean(dim=-1)).unsqueeze(-1)
        logits = self.classifier(features.transpose(1, 2)).transpose(1, 2)
        logits = F.interpolate(logits, size=x.shape[-1], mode='linear', align_corners=True)
        result = logits.mean(dim=-1)
        return result.squeeze(-1) if result.shape[-1] == 1 else result

class BusiaBeatTransformer(nn.Module):

    def __init__(self, in_channels=12, n_classes=1):
        super().__init__()
        self.tokenizer = nn.Conv1d(in_channels, 16, 3, stride=3, bias=True)
        self.position = nn.Parameter(torch.empty(1, 66, 16))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.norm_attention = nn.LayerNorm(16)
        self.attention = nn.MultiheadAttention(16, 8, dropout=0.0, batch_first=True)
        self.norm_ff = nn.LayerNorm(16)
        self.ff = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 16), nn.GELU())
        self.norm_final = nn.LayerNorm(16)
        self.rr_projection = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(18, n_classes)

    def forward(self, windows, rr):
        tokens = self.tokenizer(windows).transpose(1, 2) + self.position
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ff(self.norm_ff(tokens))
        features = self.norm_final(tokens).mean(dim=1)
        return self.classifier(torch.cat([features, self.rr_projection(rr)], dim=1))

class BusiaRecordAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.core = BusiaBeatTransformer(12, 1)
        self.register_buffer('window_offsets', torch.arange(-99, 99).float() * (100.0 / 360.0))

    def forward(self, x, aux):
        record_index = aux['record_index'].long()
        position = aux['peak_position'][:, None] + self.window_offsets[None, :]
        left = position.floor().long()
        right = left + 1
        fraction = (position - left).unsqueeze(1)
        a = x[record_index[:, None], :, left.clamp(0, x.shape[-1] - 1)].transpose(1, 2)
        b = x[record_index[:, None], :, right.clamp(0, x.shape[-1] - 1)].transpose(1, 2)
        a = a * ((left >= 0) & (left < x.shape[-1])).unsqueeze(1)
        b = b * ((right >= 0) & (right < x.shape[-1])).unsqueeze(1)
        windows = a * (1 - fraction) + b * fraction
        beat_logits = self.core(windows, aux['rr']).squeeze(-1)
        sums = x.new_zeros(x.shape[0]).scatter_add(0, record_index, beat_logits)
        counts = x.new_zeros(x.shape[0]).scatter_add(0, record_index, torch.ones_like(beat_logits))
        if bool((counts == 0).any()):
            raise ValueError('Every record must have at least one beat or its declared fallback')
        return sums / counts

def make_model(name):
    if name == 'afficient_419':
        return ECGResNet(base=1)
    if name == 'smaller_238':
        return ECGResNet(base=1, channels=(1, 1, 1))
    if name == 'basso_ph_multiscopic':
        return BassoPHMultiScopic()
    if name == 'busia_transformer':
        return BusiaRecordAdapter()
    raise ValueError(f'Unknown model: {name}')

def forward_record(model, x, aux):
    return model(x, aux) if isinstance(model, (BusiaRecordAdapter, BassoPHMultiScopic)) else model(x)

def parameter_table(model):
    rows = []
    for name, module in model.named_modules():
        for tensor_name, tensor in module.named_parameters(recurse=False):
            rows.append({'layer': name or '<root>', 'type': type(module).__name__, 'tensor': tensor_name, 'shape': 'x'.join(map(str, tensor.shape)), 'parameters': tensor.numel(), 'requires_grad': tensor.requires_grad, 'used_in_forward': not (isinstance(module, PHConv1d) and tensor_name == 'bias')})
    if sum((row['parameters'] for row in rows)) != sum((p.numel() for p in model.parameters())):
        raise AssertionError('Parameter table and unique model parameter count differ')
    return rows
