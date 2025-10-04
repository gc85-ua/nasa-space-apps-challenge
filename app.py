# app.py
import os
from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# --- Rutas robustas (relativas al archivo) ---
BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
DATA = BASE / "data"
CSV = DATA / "demo_cases.csv"
ASSETS.mkdir(exist_ok=True, parents=True)
DATA.mkdir(exist_ok=True, parents=True)

# --- Utilidades para crear la demo si falta ---
def _synth_lightcurve(period, depth_ppm, duration_hr, snr, n=1500, cleaned=False, seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, period * 4, n)
    flux = np.ones_like(time)
    depth = depth_ppm / 1e6
    width_days = duration_hr / 24.0
    for k in range(4):
        center = period * (k + 0.5)
        in_transit = np.abs(time - center) < (width_days / 2)
        flux[in_transit] -= depth
    noise = rng.normal(0, (1.0 / max(snr, 1e-6)), size=n)
    y = flux + noise
    if cleaned:
        window = max(5, n // 80)
        kernel = np.ones(window) / window
        smooth = np.convolve(y, kernel, mode="same")
        y = y / smooth
    return time, y

def _save_curve_png(path: Path, period, depth_ppm, duration_hr, snr, cleaned, seed):
    t, y = _synth_lightcurve(period, depth_ppm, duration_hr, snr, cleaned=cleaned, seed=seed)
    plt.figure(figsize=(6, 3))
    plt.plot(t, y, linewidth=1)
    plt.xlabel("Días")
    plt.ylabel("Flujo normalizado")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def ensure_demo_files():
    # CSV
    if not CSV.exists():
        df = pd.DataFrame([
            {"id":"TOI-1234","period_days":3.52,"depth_ppm":600,"duration_hr":2.1,"snr":9.8,
             "odd_even_ok":1,"secondary_ok":1,"centroid_ok":1,"score":0.87},
            {"id":"TOI-9999","period_days":0.85,"depth_ppm":8500,"duration_hr":1.0,"snr":6.1,
             "odd_even_ok":0,"secondary_ok":0,"centroid_ok":0,"score":0.12},
        ])
        df.to_csv(CSV, index=False)
    # Imágenes
    df = pd.read_csv(CSV)
    for row in df.itertuples(index=False):
        raw_path = ASSETS / f"{row.id}_raw.png"
        clean_path = ASSETS / f"{row.id}_clean.png"
        if not raw_path.exists():
            _save_curve_png(raw_path, row.period_days, row.depth_ppm, row.duration_hr, row.snr, cleaned=False, seed=42)
        if not clean_path.exists():
            _save_curve_png(clean_path, row.period_days, row.depth_ppm, row.duration_hr, row.snr, cleaned=True, seed=42)

ensure_demo_files()

# --- Backend (sustituye aquí por tu API real cuando quieras) ---
def get_case_data(object_id: str) -> dict:
    if not CSV.exists():
        raise FileNotFoundError(f"No existe el CSV de demo en: {CSV}")
    df = pd.read_csv(CSV)
    if object_id not in set(df["id"]):
        raise ValueError(f"ID '{object_id}' no está en {CSV.name}. Prueba TOI-1234 o TOI-9999.")
    row = df[df["id"] == object_id].iloc[0]
    return dict(
        id=row["id"],
        period_days=float(row["period_days"]),
        depth_ppm=float(row["depth_ppm"]),
        duration_hr=float(row["duration_hr"]),
        snr=float(row["snr"]),
        odd_even_ok=bool(row["odd_even_ok"]),
        secondary_ok=bool(row["secondary_ok"]),
        centroid_ok=bool(row["centroid_ok"]),
        score=float(row["score"]),
        img_raw=str(ASSETS / f"{row['id']}_raw.png"),
        img_clean=str(ASSETS / f"{row['id']}_clean.png"),
    )

def predict_backend(object_id: str, threshold: float = 0.5) -> dict:
    d = get_case_data(object_id)
    score = d["score"]
    label = "Candidato" if score >= threshold else "Dudoso/No planeta"
    explain = {
        "Periodo": f"{d['period_days']:.2f} d",
        "Profundidad": f"{int(d['depth_ppm'])} ppm",
        "Odd/Even": "OK" if d["odd_even_ok"] else "⚠️",
        "Secundaria": "NO" if d["secondary_ok"] else "✖️",
        "Centroides": "Estables" if d["centroid_ok"] else "Inestables",
    }
    return dict(score=score, label=label, explain=explain, data=d)

def chips_markdown(explain: dict) -> str:
    chips = [
        f"`Periodo {explain['Periodo']}`",
        f"`Profundidad {explain['Profundidad']}`",
        f"`Odd/Even {explain['Odd/Even']}`",
        f"`Secundaria {explain['Secundaria']}`",
        f"`Centroides {explain['Centroides']}`",
    ]
    return " · ".join(chips)

def generate_report(object_id: str, score: float, explain: dict, out_dir: Path = BASE) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = out_dir / f"reporte_{object_id}_{stamp}.html"
    html = f"""
    <html>
    <head><meta charset='utf-8'><title>Reporte {object_id}</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 720px; margin: 24px auto;">
      <h2>Resultado — {object_id}</h2>
      <h1 style="font-size: 42px; margin: 0 0 16px 0;">Prob. planeta: {int(round(score*100))}%</h1>
      <p style="opacity:.8;">Generado: {dt.datetime.now().isoformat(timespec='seconds')}</p>
      <h3>Señales que miramos</h3>
      <ul>
        <li>Periodo: {explain['Periodo']}</li>
        <li>Profundidad: {explain['Profundidad']}</li>
        <li>Odd/Even: {explain['Odd/Even']}</li>
        <li>Secundaria: {explain['Secundaria']}</li>
        <li>Centroides: {explain['Centroides']}</li>
      </ul>
      <p style="font-size:12px; opacity:.7;">Nota: Demo de prototipo. Resultados orientativos.</p>
    </body>
    </html>
    """
    fpath.write_text(html, encoding="utf-8")
    return str(fpath)

# --- UI Gradio ---
with gr.Blocks(title="Exoplanet Demo") as demo:
    gr.Markdown("# ¿Hay un planeta aquí?\nSelecciona una estrella de ejemplo y analiza la señal.")
    with gr.Row():
        object_id = gr.Dropdown(choices=["TOI-1234", "TOI-9999"], value="TOI-1234", label="ID (TOI/TIC)", interactive=True)
        threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Umbral (¿y si…?)")
        analyze_btn = gr.Button("Analizar", variant="primary")
    with gr.Row():
        img_raw = gr.Image(type="filepath", label="Curva — antes")
        img_clean = gr.Image(type="filepath", label="Curva — después")
    score_md = gr.Markdown("**Probabilidad de planeta:** —")
    chips_md = gr.Markdown("Señales que miramos: —")
    with gr.Row():
        report_btn = gr.Button("Generar informe")
        report_file = gr.File(label="Descargar informe", interactive=False)

    def on_analyze(object_id, threshold):
        try:
            pred = predict_backend(object_id, threshold)
            d = pred["data"]
            score = pred["score"]
            label = pred["label"]
            md = f"**Probabilidad de planeta:** **{int(round(score*100))}%** — _{label}_"
            return d["img_raw"], d["img_clean"], md, chips_markdown(pred["explain"])
        except Exception as e:
            return None, None, f"⚠️ Error: {e}", "—"

    def on_report(object_id, threshold):
        pred = predict_backend(object_id, threshold)
        fpath = generate_report(object_id, pred["score"], pred["explain"], out_dir=BASE)
        return fpath

    analyze_btn.click(on_analyze, [object_id, threshold], [img_raw, img_clean, score_md, chips_md])
    report_btn.click(on_report, [object_id, threshold], [report_file])

if __name__ == "__main__":
    demo.launch()  # añade share=True si quieres link público
