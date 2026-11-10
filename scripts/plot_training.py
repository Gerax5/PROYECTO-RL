"""Gráficas de entrenamiento a partir de history.json (una o varias corridas).

Uso:
    python scripts/plot_training.py runs/ppo runs/rppo_curr --out runs/training_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica en orden fijo (una corrida = un color, siempre el mismo).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

PANELS = [
    ("f1_counters", "F1 (contadores)", 1.0, None),
    ("recall_counter", "Recall contadores", 1.0, None),
    ("recall_stealth", "Recall disimulados", 1.0, None),
    ("fp_rate", "Tasa de falsos positivos", 1.0, 0.05),
    ("detection_rounds", "Rondas hasta detección", None, None),
    ("reward_per_100", "Recompensa / 100 rondas", None, None),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/training_curves.png"))
    args = p.parse_args()

    histories = {r.name: json.loads((r / "history.json").read_text()) for r in args.runs}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title, pct, limit) in zip(axes.flat, PANELS):
        for color, (name, hist) in zip(COLORS, histories.items()):
            x = [h["timesteps"] / 1e6 for h in hist]
            y = [h[key] * (100 if pct else 1) for h in hist]
            ax.plot(x, y, color=color, linewidth=2, label=name)
        if limit is not None:
            ax.axhline(limit * 100, color=MUTED, linestyle="--", linewidth=1)
            ax.text(0, limit * 100, " máx. permitido", color=MUTED, fontsize=8, va="bottom")
        ax.set_title(title + (" (%)" if pct else ""), color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Pasos (millones)", color=MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Supervisor: métricas de validación durante el entrenamiento", color=INK, fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor="white")
    print(f"Gráfica guardada en {args.out}")


if __name__ == "__main__":
    main()
