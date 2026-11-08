import json
import re

import numpy as np

from env import Table
from env.casino_env import ACTION_EXPEL
from eval.replay import record_episode, write_html
from supervisor import HeuristicSupervisor, NeverExpel


class ExpelSeat0:
    def reset(self): ...

    def act(self, obs):
        a = np.zeros(len(obs["seats"]), dtype=int)
        a[0] = ACTION_EXPEL
        return a


def test_table_keeps_last_round_cards():
    table = Table(rng=np.random.default_rng(0))
    table.fill_initial_seats()
    table.play_round()
    assert table.last_round is not None
    assert len(table.last_round.dealer.cards) >= 2


def test_record_episode_structure():
    replay = record_episode(HeuristicSupervisor(), seed=3, rounds=40)
    frames = replay["frames"]
    assert len(frames) == 40
    assert [f["round"] for f in frames] == list(range(1, 41))
    assert all(len(f["seats"]) == replay["meta"]["seats"] for f in frames)
    assert any(f["dealer"] and len(f["dealer"]["cards"]) >= 2 for f in frames)
    # Quienes se sentaron en reset() aparecen en el primer cuadro
    assert any(e["kind"] == "sit" for e in frames[0]["log"])
    seat = next(s for f in frames for s in f["seats"] if s.get("features"))
    assert set(seat["features"]) == set(replay["features"])


def test_record_counts_expulsions():
    replay = record_episode(ExpelSeat0(), seed=1, rounds=30)
    last = replay["frames"][-1]
    expelled = [e for f in replay["frames"] for e in f["log"] if e["kind"] == "expelled"]
    assert expelled and all(e["seat"] == 0 for e in expelled)
    assert last["hits"] + last["false_positives"] == len(expelled)
    assert sum(not e["correct"] for e in expelled) == last["false_positives"]


def test_write_html_embeds_parseable_data(tmp_path):
    replay = record_episode(NeverExpel(), seed=2, rounds=10, meta={"policy": "</script> prueba"})
    path = write_html(replay, tmp_path / "sub" / "mesa.html")
    html = path.read_text(encoding="utf-8")
    assert "__REPLAY_DATA__" not in html
    raw = re.search(r'type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(raw)["meta"]["policy"] == "</script> prueba"
