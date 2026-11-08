"""Adversario evolutivo: CMA-ES busca el camuflaje que más gana sin ser expulsado.

La aptitud combina dos mediciones de cada candidato:
    - EV: simulación sin supervisor, con muchas rondas porque la señal es pequeña.
    - Tasa de expulsión: episodios contra el supervisor actual.

    aptitud = EV (u/100 rondas) - expelled_penalty * tasa de expulsión

Con la fórmula original `ev_per_round - 50 * expelled`, el EV (~0.01 por ronda) no pesaba
nada: ganaba cualquier camuflaje no expulsado, aunque perdiera dinero.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, replace
from typing import Callable, Protocol

import cma
import numpy as np

from env import CasinoTableEnv, SeatEvent, TableConfig
from env.casino_env import ACTION_EXPEL, ACTION_NONE
from players import StealthConfig, StealthCounterPlayer, simulate_table

# Límites de cada campo de StealthConfig, en el mismo orden.
BOUNDS = (
    (2.0, 12.0),    # bet_spread_max
    (0.0, 0.6),     # bet_noise_std
    (0.0, 0.5),     # cover_bet_prob
    (0.0, 1.0),     # index_play_prob
    (0.0, 1.0),     # deliberate_error_prob
    (0.0, 3.0),     # delay_bet_ramp_rounds
)


def decode(genome: np.ndarray) -> StealthConfig:
    """Genoma en [0, 1]^6 -> camuflaje; el retraso se redondea porque es entero."""
    values = [lo + float(g) * (hi - lo) for g, (lo, hi) in zip(np.clip(genome, 0, 1), BOUNDS)]
    values[-1] = round(values[-1])
    return StealthConfig(*values)


def encode(config: StealthConfig) -> np.ndarray:
    return np.array([(v - lo) / (hi - lo) for v, (lo, hi) in zip(astuple(config), BOUNDS)])


class Supervisor(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: dict[str, np.ndarray], events: list[SeatEvent]) -> list[int]: ...


class CorrelationSupervisor:
    """Heurística de la sección 7.1, para evolucionar mientras no haya política entrenada."""

    def __init__(self, threshold: float = 0.6, min_rounds: int = 30, window: int = 50, seats: int = 5):
        self.threshold, self.min_rounds, self.window, self.seats = threshold, min_rounds, window, seats
        self.reset()

    def reset(self) -> None:
        self._history: list[list[tuple[float, float]]] = [[] for _ in range(self.seats)]

    def act(self, obs, events) -> list[int]:
        for e in events:
            if e.event == "play" and e.bet > 0:
                self._history[e.seat].append((e.bet, e.true_count))
                del self._history[e.seat][: -self.window]
            elif e.event != "play":
                self._history[e.seat] = []
        return [ACTION_EXPEL if self._is_suspect(h) else ACTION_NONE for h in self._history]

    def _is_suspect(self, history: list[tuple[float, float]]) -> bool:
        if len(history) < self.min_rounds:
            return False
        bets, tcs = np.array(history).T
        if bets.std() == 0 or tcs.std() == 0:
            return False
        return float(np.corrcoef(bets, tcs)[0, 1]) > self.threshold


@dataclass(frozen=True)
class AdversaryConfig:
    population: int = 32
    generations: int = 10
    sigma0: float = 0.2
    expelled_penalty: float = 5.0       # u/100: ser siempre expulsado equivale a perder 5 u/100
    ev_rounds: int = 200_000            # error estándar cercano a 0.4 u/100
    episodes: int = 4
    rounds_per_episode: int = 600
    population_mix: tuple[tuple[str, float], ...] = (("casual", 0.5), ("stealth", 0.5))


@dataclass(frozen=True)
class Evaluation:
    stealth: StealthConfig
    ev_per_100: float
    expel_rate: float
    fitness: float


def measure_expel_rate(
    stealth: StealthConfig,
    supervisor: Supervisor,
    cfg: AdversaryConfig,
    seed: int,
) -> float:
    """Fracción de disimulados, entre los que llegaron a sentarse, que fue expulsada."""
    table = TableConfig(population_mix=dict(cfg.population_mix), stealth=stealth)
    env = CasinoTableEnv(table, rounds_per_episode=cfg.rounds_per_episode)
    sat = expelled = 0
    for episode in range(cfg.episodes):
        supervisor.reset()
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            events = info["events"]
            sat += sum(e.event == "sit" and e.true_type == "stealth" for e in events)
            obs, _, _, done, info = env.step(supervisor.act(obs, events))
            expelled += sum(x["true_type"] == "stealth" for x in info["expulsions"])
    return expelled / sat if sat else 0.0


def evaluate(
    stealth: StealthConfig,
    make_supervisor: Callable[[], Supervisor],
    cfg: AdversaryConfig,
    seed: int,
) -> Evaluation:
    player = StealthCounterPlayer(stealth, rng=np.random.default_rng(seed))
    ev = simulate_table([player], cfg.ev_rounds, seed).pop().units_per_100
    expel_rate = measure_expel_rate(stealth, make_supervisor(), cfg, seed)
    return Evaluation(stealth, ev, expel_rate, ev - cfg.expelled_penalty * expel_rate)


class League:
    """Adversarios de versiones anteriores, para que el supervisor no olvide detectarlos."""

    def __init__(self, size: int = 10):
        self.size = size
        self.members: list[StealthConfig] = []

    def add(self, stealth: StealthConfig) -> None:
        self.members.append(stealth)
        del self.members[: -self.size]

    def table_config(self, base: TableConfig | None = None) -> TableConfig:
        """Mesa donde cada disimulado que llega sale de la liga."""
        return replace(base if base is not None else TableConfig(), stealth_pool=tuple(self.members))


class Evolver:
    """Conserva la búsqueda entre ciclos: cada ciclo parte del mejor camuflaje anterior."""

    def __init__(
        self,
        make_supervisor: Callable[[], Supervisor],
        cfg: AdversaryConfig | None = None,
        start: StealthConfig | None = None,
        seed: int = 0,
        league: League | None = None,
    ):
        self.make_supervisor = make_supervisor
        self.cfg = cfg if cfg is not None else AdversaryConfig()
        self.seed = seed
        self.league = league if league is not None else League()
        self.best = start if start is not None else StealthConfig()
        self.history: list[Evaluation] = []
        self._cycle = 0

    def run_cycle(self, map_fn=map) -> Evaluation:
        """Evoluciona contra el supervisor actual y añade el resultado a la liga.

        `map_fn` permite paralelizar (`pool.map`), con `make_supervisor` serializable.
        El resultado se reevalúa con otra semilla: la puntuación de la generación
        estaría inflada por haber elegido al candidato con mejor suerte.
        """
        cfg = self.cfg
        self._cycle += 1
        base_seed = self.seed + 1_000_000 * self._cycle
        es = cma.CMAEvolutionStrategy(
            encode(self.best),
            cfg.sigma0,
            {"bounds": [0, 1], "popsize": cfg.population, "seed": base_seed % (2**31 - 1) + 1, "verbose": -9},
        )
        for generation in range(cfg.generations):
            genomes = es.ask()
            # Misma semilla en la generación: todos los candidatos ven las mismas cartas.
            seed = base_seed + generation
            jobs = [(decode(g), self.make_supervisor, cfg, seed) for g in genomes]
            results = list(map_fn(_evaluate_job, jobs))
            es.tell(genomes, [-r.fitness for r in results])

        final = decode(es.result.xfavorite)
        result = evaluate(final, self.make_supervisor, cfg, base_seed + 500_000)
        self.best = final
        self.league.add(final)
        self.history.append(result)
        return result


def _evaluate_job(job: tuple) -> Evaluation:
    return evaluate(*job)
