import ast
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from afficient_dx.constants import LEADS

def label_afib(value):
    codes = ast.literal_eval(value)
    if not isinstance(codes, dict):
        raise ValueError('scp_codes must contain a Python dictionary')
    return float('AFIB' in codes)

def require_two_classes(y, name):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f'{name} must contain both AF and non-AF records')

def load_metadata(data_dir):
    df = pd.read_csv(data_dir / 'ptbxl_database.csv', index_col='ecg_id')
    required = {'scp_codes', 'patient_id', 'strat_fold', 'filename_lr'}
    if not required.issubset(df.columns) or not df.index.is_unique:
        raise ValueError('Missing PTB-XL metadata columns or duplicate ecg_id')
    if df[list(required)].isna().any().any():
        raise ValueError('Required PTB-XL metadata contains missing values')
    if not df.strat_fold.isin(range(1, 11)).all():
        raise ValueError('strat_fold must be an integer in 1..10')
    df['target'] = df.scp_codes.apply(label_afib).astype(np.float32)
    splits = [df[df.strat_fold <= 8].copy(), df[df.strat_fold == 9].copy(), df[df.strat_fold == 10].copy()]
    patients = [set(part.patient_id) for part in splits]
    if any((patients[i] & patients[j] for i in range(3) for j in range(i))):
        raise ValueError('Patient leakage across official PTB-XL splits')
    for name, part in zip(['training', 'validation', 'test'], splits):
        require_two_classes(part.target, name)
    return tuple(splits)

def partition_validation(df, seed):
    groups = df.groupby('patient_id', sort=True).target.max()
    try:
        selection_ids, other_ids = train_test_split(groups.index.to_numpy(), test_size=0.5, random_state=seed, stratify=groups.to_numpy())
        calibration_ids, threshold_ids = train_test_split(other_ids, test_size=0.5, random_state=seed + 1, stratify=groups.loc[other_ids].to_numpy())
    except ValueError as exc:
        raise ValueError('Too few patients/classes to split fold 9 safely') from exc
    out = {name: df[df.patient_id.isin(ids)].copy() for name, ids in zip(['selection', 'calibration', 'threshold'], [selection_ids, calibration_ids, threshold_ids])}
    for name, part in out.items():
        require_two_classes(part.target, name)
    return out

def canonical_leads(names):
    if names == ['all']:
        return LEADS.copy()
    lookup = {lead.upper(): lead for lead in LEADS}
    try:
        result = [lookup[name.upper()] for name in names]
    except KeyError as exc:
        raise ValueError(f'Unknown ECG lead: {exc.args[0]}') from exc
    if not result or len(result) != len(set(result)):
        raise ValueError('Leads must be nonempty and unique')
    return result

def load_record(path, frequency, leads=None):
    import wfdb
    leads = LEADS if leads is None else leads
    x, fields = wfdb.rdsamp(str(path))
    if float(fields['fs']) != 100.0:
        raise ValueError(f'{path}: filename_lr must be the original 100-Hz record')
    names = [s.upper() for s in fields['sig_name']]
    order = [names.index(s.upper()) for s in leads]
    x = x[:, order].T.astype(np.float32)
    if x.shape[1] != 1000 or not np.isfinite(x).all():
        raise ValueError(f'{path}: expected finite 10-second ECG')
    if frequency == 50:
        x = resample_poly(x, 1, 2, axis=1, padtype='line').astype(np.float32)
    elif frequency != 100:
        raise ValueError('Supported sampling rates are 50 and 100 Hz')
    return x

def load_split(df, data_dir, frequency, leads=None):
    x = np.stack([load_record(data_dir / f, frequency, leads) for f in df.filename_lr])
    return (x, df.target.to_numpy(np.float32))
