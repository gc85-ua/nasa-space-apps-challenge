
# (same content as before, re-inserted)
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union
import json, re, math
from pathlib import Path

import pandas as pd
import numpy as np

def days_to_hours(x): return x * 24.0
def hours_to_days(x): return x / 24.0
def minutes_to_hours(x): return x / 60.0
def hours_to_minutes(x): return x * 60.0

def ppm_to_frac(x): return x / 1e6
def frac_to_ppm(x): return x * 1e6
def percent_to_ppm(x): return x * 1e4
def ppm_to_percent(x): return x / 1e4

def btjd_to_bjd(x, offset=2457000.0): return x + offset
def bjd_to_btjd(x, offset=2457000.0): return x - offset

def rjup_to_rearth(x): return x * 11.209
def rearth_to_rjup(x): return x / 11.209

def rsun_to_rearth(x): return x * 109.1
def rearth_to_rsun(x): return x / 109.1

def arcsec_to_mas(x): return x * 1000.0
def mas_to_arcsec(x): return x / 1000.0

def si_g_to_logg_cgs(g_mps2):
    g = np.array(g_mps2, dtype=float)
    g[g <= 0] = np.nan
    return np.log10(g * 100.0)

def logg_cgs_to_si_g(logg):
    return (10.0 ** np.array(logg, dtype=float)) / 100.0

import re
_hms_re = re.compile(r"^\s*([0-9]+)h\s*([0-9]+)m\s*([0-9\.]+)s\s*$|^\s*([0-9]+):([0-9]+):([0-9\.]+)\s*$", re.I)
_dms_re = re.compile(r"^\s*([+\-]?[0-9]+)d\s*([0-9]+)m\s*([0-9\.]+)s\s*$|^\s*([+\-]?[0-9]+):([0-9]+):([0-9\.]+)\s*$", re.I)

def _parse_hms_to_hours(s: str):
    m = _hms_re.match(str(s))
    if not m: return np.nan
    if m.group(1):
        h, mnt, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
    else:
        h, mnt, sec = float(m.group(4)), float(m.group(5)), float(m.group(6))
    return h + mnt/60.0 + sec/3600.0

def _parse_dms_to_deg(s: str):
    m = _dms_re.match(str(s))
    if not m: return np.nan
    if m.group(1):
        d, mnt, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
    else:
        d, mnt, sec = float(m.group(4)), float(m.group(5)), float(m.group(6))
    sign = -1.0 if d < 0 else 1.0
    d = abs(d)
    return sign * (d + mnt/60.0 + sec/3600.0)

def hms_to_deg(x):
    if pd.isna(x): return np.nan
    h = _parse_hms_to_hours(x)
    return np.nan if pd.isna(h) else h * 15.0

def dms_to_deg(x):
    if pd.isna(x): return np.nan
    d = _parse_dms_to_deg(x)
    return np.nan if pd.isna(d) else d

CONVERTERS = {
    "days_to_hours": days_to_hours,
    "hours_to_days": hours_to_days,
    "minutes_to_hours": minutes_to_hours,
    "hours_to_minutes": hours_to_minutes,
    "ppm_to_frac": ppm_to_frac,
    "frac_to_ppm": frac_to_ppm,
    "percent_to_ppm": percent_to_ppm,
    "ppm_to_percent": ppm_to_percent,
    "btjd_to_bjd": btjd_to_bjd,
    "bjd_to_btjd": bjd_to_btjd,
    "rjup_to_rearth": rjup_to_rearth,
    "rearth_to_rjup": rearth_to_rjup,
    "rsun_to_rearth": rsun_to_rearth,
    "rearth_to_rsun": rearth_to_rsun,
    "arcsec_to_mas": arcsec_to_mas,
    "mas_to_arcsec": mas_to_arcsec,
    "si_g_to_logg_cgs": si_g_to_logg_cgs,
    "logg_cgs_to_si_g": logg_cgs_to_si_g,
    "hms_to_deg": hms_to_deg,
    "dms_to_deg": dms_to_deg,
    "scale": lambda x, factor=1.0: x * factor,
    "offset": lambda x, amount=0.0: x + amount,
}

from dataclasses import dataclass
@dataclass
class CleanReport:
    steps: list

    def add(self, **kwargs):
        self.steps.append(kwargs)

    def to_dataframe(self):
        return pd.DataFrame(self.steps)

def _coerce_numeric(series: pd.Series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

def apply_conversion(series: pd.Series, spec, report, colname: str) -> pd.Series:
    before_nonnull = int(series.notna().sum())
    s = series.copy()
    if isinstance(spec, str):
        func = CONVERTERS.get(spec)
        if func is None:
            raise ValueError(f"Conversor '{spec}' no existe.")
        s = _coerce_numeric(s) if spec not in ("hms_to_deg","dms_to_deg") else s
        out = func(s)
        used = {"method": spec}
    elif isinstance(spec, dict):
        method = spec.get("method")
        if method in ("scale","offset"):
            func = CONVERTERS[method]
            s = _coerce_numeric(s)
            if method == "scale":
                out = func(s, factor=float(spec.get("factor", 1.0)))
            else:
                out = func(s, amount=float(spec.get("amount", 0.0)))
        else:
            func = CONVERTERS.get(method)
            if func is None:
                raise ValueError(f"Conversor '{method}' no existe.")
            s = _coerce_numeric(s) if method not in ("hms_to_deg","dms_to_deg") else s
            kwargs = {k:v for k,v in spec.items() if k not in ("method","new_name")}
            out = func(s, **kwargs)
        used = spec
    elif callable(spec):
        out = spec(s)
        used = {"callable": getattr(spec, "__name__", str(spec))}
    else:
        raise ValueError(f"Spec no válido: {spec}")
    after_nonnull = int(pd.Series(out).notna().sum())
    report.add(op="convert", column=colname, spec=json.dumps(used), nonnull_before=before_nonnull, nonnull_after=after_nonnull)
    return pd.Series(out, index=series.index)

def sanitize_column_names(cols):
    cleaned = []
    seen = {}
    for c in cols:
        new = re.sub(r"\s+", "_", str(c).strip())
        new = re.sub(r"[^0-9a-zA-Z_]+", "_", new)
        new = re.sub(r"_+", "_", new).strip("_").lower()
        base = new or "col"
        idx = seen.get(base, 0)
        if idx > 0:
            new = f"{base}_{idx}"
        seen[base] = idx + 1
        cleaned.append(new)
    return cleaned

def data_clean(input_csv, output_csv, rename_map=None, convert_map=None, keep_original=False, create_report=True, report_csv=None, encoding="utf-8", sep=None):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    report = CleanReport(steps=[])

    if sep is None:
        df = pd.read_csv(input_csv, engine="python", comment="#")
    else:
        df = pd.read_csv(input_csv, engine="python", comment="#", sep=sep)

    df.columns = [str(c).strip() for c in df.columns]

    if rename_map:
        df = df.rename(columns=rename_map)
        for old, new in rename_map.items():
            report.add(op="rename", column=old, spec=json.dumps({"new_name": new}))

    if convert_map:
        for col, spec in convert_map.items():
            if col not in df.columns:
                report.add(op="warning", column=col, spec="missing_column")
                continue
            series = df[col]
            new_series = apply_conversion(series, spec, report, col)
            if isinstance(spec, dict) and "new_name" in spec:
                new_name = spec["new_name"]
            else:
                new_name = f"{col}_std" if keep_original else col
            if keep_original and new_name == col:
                new_name = f"{col}_std"
            df[new_name] = new_series
            if not keep_original and new_name == col:
                df[col] = new_series

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=encoding)

    if create_report:
        rep_path = Path(report_csv) if report_csv else output_csv.with_suffix(".clean_report.csv")
        pd.DataFrame(report.steps).to_csv(rep_path, index=False, encoding=encoding)

    return output_csv

def _load_json(path):
    if not path: return None
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Limpia y estandariza un CSV (renombrado + unidades).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rename-map")
    ap.add_argument("--convert-map")
    ap.add_argument("--keep-original", action="store_true")
    ap.add_argument("--sep")
    args = ap.parse_args()

    rename_map = _load_json(args.rename_map)
    convert_map = _load_json(args.convert_map)

    out = data_clean(
        input_csv=args.input,
        output_csv=args.output,
        rename_map=rename_map,
        convert_map=convert_map,
        keep_original=args.keep_original,
        sep=args.sep,
    )
    print(f"OK -> {out}")

if __name__ == "__main__":
    main()
