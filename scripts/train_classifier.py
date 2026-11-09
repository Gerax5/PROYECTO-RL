"""Entrena el baseline supervisado (regresión logística y gradient boosting).

Los datos de entrenamiento y de prueba salen de semillas distintas.

Uso:
    python scripts/train_classifier.py --episodes 60
    python scripts/evaluate_baselines.py --classifier models/classifier_gboost.joblib
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from env import TableConfig  # noqa: E402
from supervisor.classifier import KINDS, ClassifierSupervisor, collect_samples, fit_classifier  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60, help="episodios de entrenamiento")
    p.add_argument("--test-episodes", type=int, default=15)
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--session-mean", type=float, default=100.0,
                   help="duración media de sesión igual para todos (0 = la de TableConfig)")
    p.add_argument("--out", type=Path, default=ROOT / "models")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = TableConfig()
    if args.session_mean > 0:
        cfg = replace(cfg, session_mean_rounds={k: args.session_mean for k in cfg.population_mix})
    train_seeds = list(range(args.seed, args.seed + args.episodes))
    test_seeds = list(range(args.seed + args.episodes, args.seed + args.episodes + args.test_episodes))

    x, y = collect_samples(train_seeds, cfg, args.rounds)
    x_test, y_test = collect_samples(test_seeds, cfg, args.rounds)
    print(f"{len(y):,} muestras de entrenamiento ({y.mean():.0%} contadores), {len(y_test):,} de prueba")

    for kind in KINDS:
        model = fit_classifier(x, y, kind)
        auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        path = args.out / f"classifier_{kind}.joblib"
        ClassifierSupervisor(model).save(path)
        print(f"{kind:9s} AUC de prueba {auc:.3f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
