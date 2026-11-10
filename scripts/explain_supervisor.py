"""Explicación global del supervisor (XAI): qué features pesan en sus decisiones.

Juega varios episodios, explica cada decisión con `xai.make_explainer` y promedia las
atribuciones por grupo:
    expulsion_correcta   expulsiones de contadores (abiertos o disimulados)
    falso_positivo       expulsiones de casuales
    no_expulsado_<tipo>  jugadores sentados que no se expulsaron en esa ronda
Con un clasificador añade la importancia por permutación (caída de AUC).

Uso:
    python scripts/explain_supervisor.py --model runs/rppo --episodes 5
    python scripts/explain_supervisor.py --classifier models/classifier_gboost.joblib --threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from env import CasinoTableEnv, TableConfig  # noqa: E402
from env.casino_env import ACTION_EXPEL  # noqa: E402
from players import StealthConfig  # noqa: E402
from xai import AttributionStats, make_explainer  # noqa: E402

SHOW = 8   # features por grupo en la tabla impresa


def run(policy, explainer, seeds: list[int], cfg: TableConfig, rounds: int, every: int) -> AttributionStats:
    stats = AttributionStats()
    env = CasinoTableEnv(cfg, rounds_per_episode=rounds)
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        policy.reset()
        types: list[str | None] = [None] * env.table_config.seats
        step, done = 0, False
        while not done:
            for e in info["events"]:
                if e.event == "sit":
                    types[e.seat] = e.true_type
                elif e.event in ("leave", "expelled"):
                    types[e.seat] = None
            explanations = explainer.explain(obs)
            action = np.asarray(policy.act(obs))
            for seat, x in enumerate(explanations):
                if x is None or types[seat] is None:
                    continue
                if action[seat] == ACTION_EXPEL:
                    stats.add("expulsion_correcta" if types[seat] != "casual" else "falso_positivo", x["attr"])
                elif step % every == 0:
                    stats.add(f"no_expulsado_{types[seat]}", x["attr"])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
    return stats


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path, help="corrida entrenada (runs/<nombre>)")
    src.add_argument("--classifier", type=Path, help="clasificador .joblib")
    p.add_argument("--which", default="best", help="checkpoint de la corrida: best | final | last")
    p.add_argument("--threshold", type=float, default=0.9, help="umbral de P(contador) del clasificador")
    p.add_argument("--min-rounds", type=int, default=30)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--every", type=int, default=10, help="muestrear decisiones de 'no expulsar' cada N rondas")
    p.add_argument("--disguise", type=float, default=None)
    p.add_argument("--out", type=Path, default=None, help="JSON con los resultados")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = TableConfig()
    if args.disguise is not None:
        cfg = replace(cfg, stealth=StealthConfig.from_disguise(args.disguise))
    if args.model is not None:
        from supervisor.train import load_supervisor

        policy = load_supervisor(args.model, args.which)
        name = args.model.name
    else:
        from supervisor.classifier import ClassifierSupervisor

        policy = ClassifierSupervisor.load(args.classifier, threshold=args.threshold, min_rounds=args.min_rounds)
        name = args.classifier.stem
    explainer = make_explainer(policy)
    print(f"{name}: {explainer.method}, {args.episodes} episodios de {args.rounds} rondas")

    seeds = list(range(args.seed, args.seed + args.episodes))
    stats = run(policy, explainer, seeds, cfg, args.rounds, args.every)
    result: dict = {"name": name, "method": explainer.method, "seeds": seeds, "groups": {}}
    for group in sorted(stats.groups()):
        table = stats.table(group)
        result["groups"][group] = {"n": stats.n[group], "features": table}
        print(f"\n{group} (n = {stats.n[group]})   media firmada | media |abs|")
        for row in table[:SHOW]:
            print(f"  {row['feature']:27s} {row['mean']:+8.3f} | {row['mean_abs']:7.3f}")

    if args.classifier is not None:
        from supervisor.classifier import collect_samples
        from xai.tabular import permutation_importance_table

        x, y = collect_samples(list(range(args.seed + 1000, args.seed + 1000 + args.episodes)), cfg, args.rounds)
        imp = permutation_importance_table(policy.model, x, y)
        result["permutation_importance"] = imp
        print("\nimportancia por permutación (caída de AUC)")
        for row in imp[:SHOW]:
            print(f"  {row['feature']:27s} {row['auc_drop']:+.4f} ± {row['std']:.4f}")

    out = args.out or ROOT / "runs" / f"xai_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
