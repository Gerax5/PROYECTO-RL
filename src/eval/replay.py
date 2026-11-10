"""Graba un episodio del supervisor y lo guarda como una mesa visual en HTML.

Cada cuadro es un `step` del ambiente: primero las expulsiones que decidió el
supervisor con la observación anterior y después la ronda jugada. El tipo real de
cada jugador se guarda para mostrarlo, pero la política nunca lo recibe.

Con un `explainer` (ver `xai.make_explainer`) cada silla trae además `xai`: la
explicación de la decisión que el supervisor tomó en ese cuadro, calculada con la
misma observación que recibió la política.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from blackjack.cards import card_name
from env import CasinoTableEnv, RewardConfig, TableConfig
from env.casino_env import ACTION_EXPEL, ACTION_WATCH
from env.features import SEAT_FEATURES

from .evaluate import Policy

TEMPLATE = Path(__file__).with_name("replay_template.html")


def _cards(cards: list[int]) -> list[str]:
    return ["10" if card_name(c) == "T" else card_name(c) for c in cards]


def _hand(hand) -> dict:
    return {
        "cards": _cards(hand.cards),
        "total": hand.total,
        "soft": hand.is_soft,
        "bust": hand.is_bust,
        "blackjack": hand.is_blackjack,
        "doubled": hand.doubled,
        "bet": hand.bet,
    }


def _features(row: np.ndarray) -> dict[str, float]:
    return {name: round(float(row[i]), 3) for i, name in enumerate(SEAT_FEATURES)}


def record_episode(
    policy: Policy,
    seed: int,
    table_config: TableConfig | None = None,
    reward_config: RewardConfig | None = None,
    rounds: int = 600,
    meta: dict | None = None,
    explainer=None,
) -> dict:
    """Juega un episodio y devuelve los cuadros listos para `write_html`."""
    env = CasinoTableEnv(table_config, reward_config, rounds)
    obs, info = env.reset(seed=seed)
    policy.reset()
    n_seats = env.table_config.seats

    session_net: Counter = Counter()      # ganancia por player_id en su estancia
    totals = {"house_net": 0.0, "reward": 0.0, "hits": 0, "false_positives": 0, "seen": Counter()}
    # Quienes se sientan en reset() aparecen en la bitácora del primer cuadro.
    initial_log = []
    for e in info["events"]:
        if e.event == "sit":
            totals["seen"][e.true_type] += 1
            initial_log.append({"seat": e.seat, "kind": "sit", "type": e.true_type})

    frames = []
    done = False
    while not done:
        explanations = explainer.explain(obs) if explainer is not None else None
        action = np.asarray(policy.act(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        table = env.table

        seats: list[dict] = [{"occupied": False} for _ in range(n_seats)]
        log, initial_log = initial_log, []
        for x in info["expulsions"]:
            correct = x["true_type"] != "casual"
            totals["hits" if correct else "false_positives"] += 1
            log.append({"seat": x["seat"], "kind": "expelled", "type": x["true_type"],
                        "correct": correct, "rounds_seen": x["rounds_seen"]})
        for e in info["events"]:
            s = seats[e.seat]
            if e.event == "sit":
                totals["seen"][e.true_type] += 1
                s["sat"] = True
                log.append({"seat": e.seat, "kind": "sit", "type": e.true_type})
            elif e.event == "play":
                session_net[e.player_id] += e.result_units
                s.update(occupied=True, player_id=e.player_id, type=e.true_type, bet=e.bet,
                         actions=e.actions, net=e.result_units,
                         session_net=round(session_net[e.player_id], 2),
                         deviated=e.deviated_from_basic, index_play=e.matched_index_play)
            elif e.event == "leave":
                s["left"] = True
                log.append({"seat": e.seat, "kind": "leave", "type": e.true_type,
                            "session_net": round(session_net[e.player_id], 2)})

        dealer = None
        if table.last_round is not None:
            dealer = {**_hand(table.last_round.dealer),
                      "blackjack": table.last_round.dealer_blackjack}
            for r in table.last_round.seats:
                seats[r.seat]["hands"] = [_hand(h) for h in r.hands]

        for seat, s in enumerate(seats):
            s["watched"] = bool(action[seat] == ACTION_WATCH)
            s["expel_order"] = bool(action[seat] == ACTION_EXPEL)
            if s["occupied"] and not s.get("left"):
                s["features"] = _features(obs["seats"][seat])
            # La explicación es del jugador que estaba sentado al decidir: si se fue y
            # se sentó otro, sólo se muestra cuando lo expulsaron (sello en la silla).
            if explanations is not None and explanations[seat] is not None \
                    and (not s.get("sat") or s["expel_order"]):
                s["xai"] = explanations[seat]

        totals["house_net"] += info["house_net"]
        totals["reward"] += reward
        # TC público al inicio de la ronda (el que ven los contadores al apostar)
        tc = next((e.true_count for e in info["events"] if e.event == "play"), table.tc)
        frames.append({
            "round": table.round_id,
            "tc": round(float(tc), 2),
            "penetration": round(float(table.shoe.penetration_used), 3),
            "dealer": dealer,
            "seats": seats,
            "log": log,
            "reward": round(float(reward), 3),
            "house_net": round(totals["house_net"], 2),
            "total_reward": round(totals["reward"], 2),
            "hits": totals["hits"],
            "false_positives": totals["false_positives"],
            "seen": dict(totals["seen"]),
        })

    meta = {"seed": seed, "rounds": rounds, "seats": n_seats,
            "suspicion_feature": "corr_bet_tc", **(meta or {})}
    if explainer is not None:
        meta["xai"] = {"method": explainer.method, "score_label": explainer.score_label}
    return {
        "meta": meta,
        "features": list(SEAT_FEATURES),
        "frames": frames,
    }


def write_html(replay: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "</" dentro del JSON cerraría la etiqueta <script>.
    data = json.dumps(replay, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", data),
                    encoding="utf-8")
    return path

