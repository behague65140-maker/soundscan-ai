"""
SoundScan AI — Backend API

FastAPI server for AI music detection.
Uses librosa features + trained Gradient Boosting model.
Falls back to heuristic analysis if no model is trained.

Usage:
  pip install -r requirements.txt
  python server.py
"""

import os
import tempfile
import json
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from features import extract_features, get_feature_names

MODEL_DIR = Path(__file__).parent / "models"

app = FastAPI(title="SoundScan AI", version="2.0")

# Allow requests from any origin (the HTML page)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load trained model if available ──
_model = None
_feature_names = None


def load_model():
    global _model, _feature_names
    model_path = MODEL_DIR / "detector.joblib"
    meta_path = MODEL_DIR / "metadata.json"

    if model_path.exists() and meta_path.exists():
        import joblib
        _model = joblib.load(model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        _feature_names = meta["feature_names"]
        print(f"[OK] Modèle ML chargé ({meta['n_total']} fichiers d'entraînement)")
    else:
        print("[INFO] Pas de modèle ML trouvé — mode heuristique activé")
        print(f"       Pour entraîner: python train.py")


# ── Heuristic scoring (fallback when no ML model) ──

def sigmoid(x, center, scale):
    return 1 / (1 + np.exp(-(x - center) / scale))


def heuristic_score(feats: dict) -> dict:
    """Score using the same heuristic approach as the JS version, but with librosa features."""

    pSNR = sigmoid(feats["snr_db"], 50, 6)
    pNoiseFloor = sigmoid(-np.log10(feats["noise_floor"] + 1e-9), 6.5, 0.8)
    pDynamic = sigmoid(-feats["dynamic_range_db"], -9, 2.2)

    # MFCC stability: low delta variance = AI (smooth transitions)
    delta_vars = [feats.get(f"delta_mfcc_{i}_std", 0) for i in range(13)]
    avg_delta = np.mean(delta_vars)
    pDeltaMFCC = sigmoid(-avg_delta, -1.5, 0.5)

    # Beat regularity: low CV = AI (perfect timing)
    pBeat = sigmoid(-feats["beat_regularity_cv"], -0.08, 0.03)

    # RMS variability
    pRMSVar = sigmoid(-feats["rms_cv"], -0.5, 0.12)

    # Spectral flux
    pFlux = sigmoid(-feats["spectral_flux_mean"], -0.5, 0.2)

    # Stereo
    if feats.get("is_stereo"):
        pStereo = 0.55 * sigmoid(feats["stereo_corr_mean"], 0.94, 0.025) + \
                  0.45 * sigmoid(-feats["stereo_corr_std"], -0.05, 0.015)
    else:
        pStereo = 0.5

    # Spectral flatness
    flat = feats["spectral_flatness_mean"]
    if flat < 0.02:
        pFlat = 0.8
    elif flat > 0.15:
        pFlat = 0.65
    else:
        pFlat = 0.3

    # HF content (spectral rolloff)
    rolloff = feats["spectral_rolloff_mean"]
    pHF = 0.5
    if rolloff < 3000:
        pHF = 0.75
    elif rolloff > 8000:
        pHF = 0.3

    # Weighted combination
    ai_prob = (
        0.22 * pSNR +
        0.14 * pNoiseFloor +
        0.12 * pDynamic +
        0.10 * pRMSVar +
        0.10 * pFlux +
        0.09 * pDeltaMFCC +
        0.08 * pStereo +
        0.07 * pBeat +
        0.04 * pFlat +
        0.04 * pHF
    )

    return {
        "ai_score": round(float(np.clip(ai_prob * 100, 0, 100))),
        "method": "heuristic",
        "components": {
            "snr": round(float(pSNR), 3),
            "noise_floor": round(float(pNoiseFloor), 3),
            "dynamic_range": round(float(pDynamic), 3),
            "rms_variability": round(float(pRMSVar), 3),
            "spectral_flux": round(float(pFlux), 3),
            "delta_mfcc": round(float(pDeltaMFCC), 3),
            "stereo": round(float(pStereo), 3),
            "beat_regularity": round(float(pBeat), 3),
        },
    }


# ── ML scoring ──

def ml_score(feats: dict) -> dict:
    """Score using the trained Gradient Boosting model."""
    vec = np.array([[feats.get(name, 0.0) for name in _feature_names]])

    # Replace NaN/Inf
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    prob = _model.predict_proba(vec)[0]
    ai_prob = float(prob[1])  # class 1 = AI

    return {
        "ai_score": round(ai_prob * 100),
        "method": "ml",
        "confidence": round(float(max(prob)), 3),
    }


# ── API Routes ──

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze an audio file for AI detection.
    Returns AI probability score + extracted features.
    """
    # Save uploaded file to temp
    suffix = Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract features with librosa
        feats = extract_features(tmp_path)

        # Score with ML model if available, otherwise heuristic
        if _model is not None:
            score = ml_score(feats)
            # Also include heuristic for comparison
            score["heuristic"] = heuristic_score(feats)
        else:
            score = heuristic_score(feats)

        ai_pct = score["ai_score"]
        human_pct = 100 - ai_pct

        if ai_pct >= 63:
            verdict = "Probablement IA"
        elif ai_pct <= 38:
            verdict = "Probablement Humain"
        else:
            verdict = "Incertain"

        # Build indicators
        indicators = []

        if feats["snr_db"] > 58 and feats["noise_floor"] < 1e-5:
            indicators.append({
                "name": "Plancher de bruit quasi inexistant",
                "detail": f'SNR {feats["snr_db"]:.0f} dB — signal digitalement pur',
                "cls": "bad", "tag": "IA très probable"
            })
        elif feats["snr_db"] > 48:
            indicators.append({
                "name": "Signal très propre",
                "detail": f'SNR {feats["snr_db"]:.0f} dB',
                "cls": "bad", "tag": "IA possible"
            })
        else:
            indicators.append({
                "name": "Bruit naturel détecté",
                "detail": f'SNR {feats["snr_db"]:.0f} dB — bruit ambiant cohérent',
                "cls": "good", "tag": "Humain"
            })

        if feats["beat_regularity_cv"] < 0.06:
            indicators.append({
                "name": "Rythme parfaitement quantifié",
                "detail": f'CV beat = {feats["beat_regularity_cv"]:.3f} — timing trop régulier',
                "cls": "bad", "tag": "IA probable"
            })
        elif feats["beat_regularity_cv"] > 0.15:
            indicators.append({
                "name": "Micro-timing naturel",
                "detail": f'CV beat = {feats["beat_regularity_cv"]:.3f}',
                "cls": "good", "tag": "Humain"
            })

        if feats.get("is_stereo"):
            sm, ss = feats["stereo_corr_mean"], feats["stereo_corr_std"]
            if sm > 0.96 and ss < 0.02:
                indicators.append({
                    "name": "Stéréo artificielle",
                    "detail": f"Corrélation L/R = {sm:.3f} ± {ss:.3f}",
                    "cls": "bad", "tag": "IA très probable"
                })
            elif ss > 0.06:
                indicators.append({
                    "name": "Champ stéréo naturel",
                    "detail": f"Corrélation L/R = {sm:.3f} ± {ss:.3f}",
                    "cls": "good", "tag": "Humain"
                })

        return JSONResponse({
            "verdict": verdict,
            "ai_score": ai_pct,
            "human_score": human_pct,
            "method": score["method"],
            "score_details": score,
            "indicators": indicators,
            "features": {
                "snr_db": round(feats["snr_db"], 1),
                "noise_floor": float(f'{feats["noise_floor"]:.2e}'),
                "dynamic_range_db": round(feats["dynamic_range_db"], 1),
                "tempo": round(feats["tempo"], 1),
                "beat_regularity_cv": round(feats["beat_regularity_cv"], 4),
                "rms_cv": round(feats["rms_cv"], 4),
                "spectral_flatness": round(feats["spectral_flatness_mean"], 4),
                "spectral_flux": round(feats["spectral_flux_mean"], 4),
                "stereo_corr_mean": round(feats.get("stereo_corr_mean", 0.5), 4),
                "stereo_corr_std": round(feats.get("stereo_corr_std", 0.1), 4),
                "is_stereo": feats.get("is_stereo", False),
                "harmonic_ratio": round(feats["harmonic_ratio"], 4),
            },
        })

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "method": "ml" if _model is not None else "heuristic",
    }


if __name__ == "__main__":
    load_model()
    print("\n" + "=" * 50)
    print("  SoundScan AI — Backend API")
    print("  http://localhost:8000")
    print("  POST /analyze  — analyser un fichier audio")
    print("  GET  /health   — statut du serveur")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
