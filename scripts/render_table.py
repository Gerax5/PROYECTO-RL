"""Genera la mesa visual de un episodio y la abre en el navegador.

Graba un episodio con el supervisor heurístico (mientras no haya política entrenada)
y escribe una página HTML autocontenida: cartas, apuestas, conteo, sospecha por
silla, expulsiones (acierto o falso positivo) y lo que observa el supervisor.

Uso:
    python scripts/render_table.py --seed 7 --rounds 300 --disguise 0.6
    python scripts/render_table.py --policy never --no-open
    python scripts/render_table.py --model runs/rppo        # con explicaciones (XAI)
    python scripts/render_table.py --classifier models/classifier_gboost.joblib --threshold 0.9
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import TableConfig  # noqa: E402
from eval.replay import record_episode, write_html  # noqa: E402
from players import StealthConfig  # noqa: E402
from supervisor import HeuristicSupervisor, NeverExpel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--policy", choices=("heuristic", "never"), default="heuristic")
    parser.add_argument("--threshold", type=float, default=0.6, help="umbral de corr_bet_tc")
    parser.add_argument("--min-rounds", type=int, default=30)
    parser.add_argument("--disguise", type=float, default=None,
                        help="disguise_level del disimulado (por defecto el de TableConfig)")
    parser.add_argument("--model", type=Path, default=None,
                        help="corrida entrenada; reemplaza a --policy")
    parser.add_argument("--classifier", type=Path, default=None,
                        help="clasificador .joblib; reemplaza a --policy (usa --threshold como P)")
    parser.add_argument("--no-xai", action="store_true",
                        help="no calcular explicaciones (con --model o --classifier)")
    parser.add_argument("--out", type=Path, default=ROOT / "replays" / "mesa.html")
    parser.add_argument("--no-open", action="store_true", help="no abrir el navegador")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = TableConfig()
    if args.disguise is not None:
        cfg = replace(cfg, stealth=StealthConfig.from_disguise(args.disguise))

    if args.model is not None:
        from supervisor.train import load_supervisor

        policy = load_supervisor(args.model)
        # La política elige la acción más probable: no tiene umbral fijo.
        meta = {"policy": f"Supervisor RL ({args.model.name})"}
    elif args.classifier is not None:
        from supervisor.classifier import ClassifierSupervisor

        policy = ClassifierSupervisor.load(args.classifier, threshold=args.threshold,
                                           min_rounds=args.min_rounds)
        meta = {"policy": f"Clasificador {args.classifier.stem}: P > {args.threshold} "
                          f"tras {args.min_rounds} rondas", "threshold": args.threshold}
    elif args.policy == "heuristic":
        policy = HeuristicSupervisor(args.threshold, args.min_rounds)
        meta = {"policy": f"Heurística corr_bet_tc > {args.threshold} tras {args.min_rounds} rondas",
                "threshold": args.threshold}
    else:
        policy = NeverExpel()
        meta = {"policy": "Sin supervisor (nunca expulsa)"}
    if args.disguise is not None:
        meta["disguise"] = args.disguise

    explainer = None
    if not args.no_xai:
        from xai import make_explainer

        explainer = make_explainer(policy)
    replay = record_episode(policy, args.seed, cfg, rounds=args.rounds, meta=meta, explainer=explainer)
    path = write_html(replay, args.out)
    last = replay["frames"][-1]
    print(f"{path}  ({len(replay['frames'])} rondas, {last['hits']} contadores expulsados, "
          f"{last['false_positives']} casuales expulsados)")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
