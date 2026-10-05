import copy
import numpy as np
import torch
from afficient_dx.inference import make_loader, predict_logits
from afficient_dx.metrics import apply_calibrator, fit_calibrator, threshold_from_validation
from afficient_dx.profiling import benchmark_cpu
from afficient_dx.reports import evaluate_predictions
from afficient_dx.utils import write_json

def quantize_and_evaluate(model, arrays, frames, mean, std, args, out):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    available = torch.backends.quantized.supported_engines
    engine = args.quantized_engine
    if engine == 'auto':
        engine = next((candidate for candidate in ['x86', 'fbgemm', 'onednn', 'qnnpack'] if candidate in available), None)
    if engine not in available or engine is None:
        raise RuntimeError(f'INT8 backend unavailable: {args.quantized_engine}; supported: {available}')
    original_engine = torch.backends.quantized.engine
    previous_threads = torch.get_num_threads()
    try:
        torch.backends.quantized.engine = engine
        torch.set_num_threads(args.latency_threads)
        model = copy.deepcopy(model).cpu().eval()
        train_x, train_y = arrays['train']
        example = torch.from_numpy(train_x[:1])
        prepared = prepare_fx(model, get_default_qconfig_mapping(engine), (example,))
        rng = np.random.default_rng(args.split_seed + 2003)
        idx = rng.permutation(len(train_x))[:min(args.quantization_records, len(train_x))]
        with torch.inference_mode():
            for x, _ in make_loader(train_x[idx], train_y[idx], args.batch_size):
                prepared(x)
        quantized = convert_fx(prepared).eval()
        yc, zc = predict_logits(quantized, make_loader(*arrays['calibration'], args.batch_size), torch.device('cpu'))
        calibrator = fit_calibrator(yc, zc, args.calibration)
        yh, zh = predict_logits(quantized, make_loader(*arrays['threshold'], args.batch_size), torch.device('cpu'))
        threshold = threshold_from_validation(yh, apply_calibrator(zh, calibrator), args.threshold_method, args.target_sensitivity)
        traced = torch.jit.trace(quantized, example)
        artifact = out / 'model_int8.ts'
        torch.jit.save(traced, str(artifact))
        metadata = {'status': 'measured', 'backend': engine, 'method': 'FX static post-training quantization', 'activation_calibration_split': 'train', 'activation_calibration_records': len(idx), 'probability_calibration': calibrator, 'threshold': threshold, 'serialized_torchscript_bytes': artifact.stat().st_size, 'serialized_torchscript_kB': artifact.stat().st_size / 1000, 'cpu_latency': benchmark_cpu(quantized, train_x[:1], args.latency_warmup, args.latency_repeats, args.latency_threads), 'peak_runtime_ram_bytes': None, 'power_mw': None, 'scope': 'host backend and serialized model, not MCU memory/energy', 'normalization_mean': mean.flatten().tolist(), 'normalization_std': std.flatten().tolist(), 'leads': args.leads}
        write_json(out / 'int8_inference.json', metadata)
        if 'test' in arrays:
            yt, zt = predict_logits(quantized, make_loader(*arrays['test'], args.batch_size), torch.device('cpu'))
            metadata['test'] = evaluate_predictions(out, 'int8_test', frames['test'], yt, zt, calibrator, threshold, args, bootstrap=True)
        return metadata
    finally:
        try:
            if original_engine not in ('none', 'NoQEngine'):
                torch.backends.quantized.engine = original_engine
        finally:
            torch.set_num_threads(previous_threads)
