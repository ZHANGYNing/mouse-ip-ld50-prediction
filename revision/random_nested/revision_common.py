"""Shared I/O and metric utilities for the JHM revision scripts. No training here."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def save_json(path: Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig', float_format='%.17g')
    os.replace(tmp, path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f'Required file not found: {path}')
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df.columns = [str(x).strip() for x in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f'Duplicate column names: {path}')
    return df


def finite_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        raise ValueError(f'Missing column: {column}; found {list(df.columns)}')
    a = pd.to_numeric(df[column], errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(a).all():
        bad = np.flatnonzero(~np.isfinite(a))[:10].tolist()
        raise ValueError(f'{column} has missing/nonfinite values at row indices {bad}; no rows were dropped.')
    return a


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError('Metrics require equal-length, nonempty, one-dimensional arrays.')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite values in metric input; refusing silent filtering.')
    e = p - y; ae = np.abs(e)
    sse = float(np.sum(e * e)); sst = float(np.sum((y - y.mean()) ** 2))
    # R2 is 1-SSE/SST, NOT Pearson r squared; subset mean is used for each subset.
    r2 = float(1.0 - sse / sst) if len(y) >= 2 and sst > 0 else None
    if ae.max() > 300:
        raise ValueError('Fold error would overflow; inspect target units instead of clipping.')
    fe = np.power(10.0, ae)
    return {'n':int(len(y)), 'R2':r2, 'RMSE':float(np.sqrt(sse / len(y))),
            'MAE':float(ae.mean()), 'median_fold_error':float(np.median(fe)),
            'within_2_fold_percent':float(100 * np.mean(ae <= np.log10(2))),
            'within_5_fold_percent':float(100 * np.mean(ae <= np.log10(5))),
            'within_10_fold_percent':float(100 * np.mean(ae <= 1.0)),
            'observed_y_std_ddof0':float(y.std(ddof=0)), 'SSE':sse, 'SST':sst}


def locate_data_file(name: str, data_dir: Path, required: bool = True) -> Path | None:
    """Find an exact filename; never substitute an earlier experiment or conflict copy.

    A directly specified path takes precedence. Search below data_dir only when
    that file is absent. Duplicate matches must have identical SHA256 digests.
    """
    requested = Path(name)
    direct = requested if requested.is_absolute() else Path(data_dir) / requested
    if direct.is_file():
        return direct.resolve()
    if requested.is_absolute() or len(requested.parts) != 1:
        if required:
            raise FileNotFoundError(f'Required file not found: {direct}')
        return None
    matches = []
    skip_names = {'.venv', 'venv', '__pycache__', '.git', '.idea', 'site-packages', 'node_modules'}
    for folder, dirs, filenames in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in skip_names and
                   not d.startswith('JHM_revision_nested_oof') and
                   d != 'JHM_revision_group_audit']
        if name in filenames:
            matches.append((Path(folder) / name).resolve())
    matches = sorted(set(matches), key=str)
    if not matches:
        if required:
            raise FileNotFoundError(
                f'Required file not found in {data_dir} or its subfolders: {name}. '
                'Set an absolute path in revision_config.json; do not use a similarly named old dataset.')
        return None
    hashes = {sha256_file(p) for p in matches}
    if len(hashes) > 1:
        listing = '\n'.join(f'  {p}' for p in matches)
        raise ValueError(f'Multiple DIFFERENT copies of {name} were found:\n{listing}\n'
                         'Choose the correct original file explicitly with an absolute path.')
    return matches[0]
