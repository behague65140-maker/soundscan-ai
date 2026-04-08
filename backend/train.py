"""
Training script for AI music detection model.

Usage:
  1. Put AI-generated music files in data/ai/
  2. Put human-produced music files in data/human/
  3. Run: python train.py

The trained model is saved to models/detector.joblib
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

from features import extract_features, get_feature_names

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus"}


def collect_files():
    """Collect audio files from data/ai/ and data/human/ directories."""
    ai_dir = DATA_DIR / "ai"
    human_dir = DATA_DIR / "human"

    ai_files = [
        f for f in ai_dir.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS
    ] if ai_dir.exists() else []

    human_files = [
        f for f in human_dir.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS
    ] if human_dir.exists() else []

    return ai_files, human_files


def extract_dataset(ai_files, human_files):
    """Extract features from all audio files."""
    feature_names = get_feature_names()
    X, y, filenames = [], [], []

    total = len(ai_files) + len(human_files)
    print(f"\nExtraction des features de {total} fichiers...\n")

    for i, (fpath, label) in enumerate(
        [(f, 1) for f in ai_files] + [(f, 0) for f in human_files]
    ):
        tag = "IA" if label == 1 else "Humain"
        print(f"  [{i+1}/{total}] {tag}: {fpath.name}...", end=" ", flush=True)

        try:
            feats = extract_features(str(fpath))
            vec = [feats.get(name, 0.0) for name in feature_names]

            # Skip if NaN or Inf
            if any(not np.isfinite(v) for v in vec):
                print("SKIP (valeurs invalides)")
                continue

            X.append(vec)
            y.append(label)
            filenames.append(str(fpath.name))
            print("OK")

        except Exception as e:
            print(f"ERREUR: {e}")

    return np.array(X), np.array(y), filenames, feature_names


def train_model(X, y, feature_names):
    """Train a Gradient Boosting classifier with cross-validation."""
    print(f"\n{'='*60}")
    print(f"Dataset: {len(y)} fichiers ({sum(y)} IA, {sum(y==0)} Humain)")
    print(f"Features: {X.shape[1]}")
    print(f"{'='*60}\n")

    # Pipeline: scale + gradient boosting
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        ))
    ])

    # Cross-validation
    n_splits = min(5, min(sum(y), sum(y == 0)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        print(f"Cross-validation ({n_splits}-fold):")
        print(f"  Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
        print(f"  Per fold: {', '.join(f'{s:.1%}' for s in scores)}")
    else:
        print("Pas assez de fichiers pour la cross-validation.")

    # Train on full dataset
    pipeline.fit(X, y)

    # Feature importance
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    top_k = 15
    top_idx = np.argsort(importances)[::-1][:top_k]
    print(f"\nTop {top_k} features les plus importantes:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<35s} {importances[idx]:.4f}")

    # Full training report
    y_pred = pipeline.predict(X)
    print(f"\nRapport sur le jeu d'entraînement:")
    print(classification_report(y, y_pred, target_names=["Humain", "IA"]))

    return pipeline


def main():
    ai_files, human_files = collect_files()

    if len(ai_files) == 0 or len(human_files) == 0:
        print("=" * 60)
        print("ERREUR: Fichiers audio manquants!")
        print()
        print("Pour entraîner le modèle, il faut:")
        print(f"  1. Mettre des fichiers IA dans:     {DATA_DIR / 'ai'}")
        print(f"  2. Mettre des fichiers humains dans: {DATA_DIR / 'human'}")
        print()
        print(f"  Fichiers IA trouvés:    {len(ai_files)}")
        print(f"  Fichiers Humain trouvés: {len(human_files)}")
        print()
        print("Formats supportés:", ", ".join(sorted(AUDIO_EXTENSIONS)))
        print("Minimum recommandé: 20 fichiers par catégorie")
        print("=" * 60)
        sys.exit(1)

    print(f"Fichiers trouvés: {len(ai_files)} IA, {len(human_files)} Humain")

    X, y, filenames, feature_names = extract_dataset(ai_files, human_files)

    if len(y) < 6:
        print("ERREUR: Pas assez de fichiers exploitables (minimum 6).")
        sys.exit(1)

    model = train_model(X, y, feature_names)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "detector.joblib"
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé: {model_path}")

    # Save feature names
    meta_path = MODEL_DIR / "metadata.json"
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_ai": int(sum(y)),
        "n_human": int(sum(y == 0)),
        "n_total": int(len(y)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Métadonnées sauvegardées: {meta_path}")

    print(f"\n{'='*60}")
    print("Entraînement terminé! Lancez le serveur avec:")
    print("  python server.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
