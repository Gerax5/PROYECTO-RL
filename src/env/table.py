"""Mesa de blackjack con llegadas, salidas, bankroll y expulsiones.

Las cantidades de dinero van en unidades (1 unidad = apuesta mínima). La mesa lleva
su propio conteo con la misma información pública que tendría un pit boss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from blackjack import (
    Action,
    Hand,
    Rules,
    Shoe,
    basic_strategy,
    counter_strategy,
    hi_lo_value,
    play_round,
    take_insurance,
    true_count,
)
from players import (
    BasePlayer,
    CasualConfig,
    CasualPlayer,
    CounterConfig,
    CounterPlayer,
    StealthConfig,
    StealthCounterPlayer,
)

from .events import SeatEvent


@dataclass(frozen=True)
class TableConfig:
    seats: int = 5
    max_bet_units: float = 50.0         # 500 / 10 de configs/env.yaml
    n_decks: int = 6
    penetration: float = 0.75
    burn_cards: int = 1
    rules: Rules = field(default_factory=Rules)
    arrival_rate: float = 0.15
    initial_occupancy: float = 0.9      # cerca del régimen estacionario, para no empezar vacía
    population_mix: dict[str, float] = field(
        default_factory=lambda: {"casual": 0.70, "counter": 0.15, "stealth": 0.15}
    )
    session_mean_rounds: dict[str, float] = field(
        default_factory=lambda: {"casual": 80, "counter": 120, "stealth": 100}
    )
    bankroll_units: tuple[float, float] = (50.0, 400.0)
    casual: CasualConfig = field(default_factory=CasualConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)
    stealth: StealthConfig = field(default_factory=StealthConfig)


@dataclass
class SeatedPlayer:
    player: BasePlayer
    player_id: int
    bankroll: float
    rounds_left: int
    rounds_seen: int = 0
    # Dinero comprometido en la ronda actual
    staked: float = 0.0
    deviated: bool = False
    matched_index: bool = False

    @property
    def available(self) -> float:
        return self.bankroll - self.staked


class Table:
    def __init__(self, config: TableConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config if config is not None else TableConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shoe = Shoe(
            self.config.n_decks, self.config.penetration, self.config.burn_cards, self.rng
        )
        self.seats: list[SeatedPlayer | None] = [None] * self.config.seats
        self.round_id = 0
        self.running_count = 0
        self._next_player_id = 0
        self._pending: list[SeatEvent] = []

    @property
    def tc(self) -> float:
        return true_count(self.running_count, self.shoe.decks_remaining)

    def _make_player(self, kind: str) -> BasePlayer:
        cfg, rules = self.config, self.config.rules
        if kind == "casual":
            return CasualPlayer(cfg.casual, rules, self.rng)
        if kind == "counter":
            return CounterPlayer(cfg.counter, rules, self.rng)
        if kind == "stealth":
            return StealthCounterPlayer(cfg.stealth, rules, self.rng)
        raise ValueError(f"tipo de jugador desconocido: {kind}")

    def _event(self, seat: int, sp: SeatedPlayer, event: str, **kw) -> SeatEvent:
        return SeatEvent(
            round_id=self.round_id,
            seat=seat,
            player_id=sp.player_id,
            event=event,
            bet=kw.pop("bet", 0.0),
            true_count=kw.pop("true_count", self.tc),
            shoe_penetration=kw.pop("shoe_penetration", self.shoe.penetration_used),
            true_type=sp.player.player_type,
            **kw,
        )

    def seat_player(self, seat: int, kind: str) -> None:
        if self.seats[seat] is not None:
            raise ValueError(f"la silla {seat} está ocupada")
        cfg = self.config
        player = self._make_player(kind)
        # Quien se sienta a mitad de zapato no vio las cartas anteriores: empieza en 0.
        player.on_shuffle(self.shoe.decks_remaining)
        sp = SeatedPlayer(
            player=player,
            player_id=self._next_player_id,
            bankroll=float(self.rng.uniform(*cfg.bankroll_units)),
            rounds_left=int(self.rng.geometric(1 / cfg.session_mean_rounds[kind])),
        )
        self._next_player_id += 1
        self.seats[seat] = sp
        self._pending.append(self._event(seat, sp, "sit"))

    def _arrivals(self, prob: float) -> None:
        kinds = list(self.config.population_mix)
        weights = np.array([self.config.population_mix[k] for k in kinds], dtype=float)
        for seat, sp in enumerate(self.seats):
            if sp is None and self.rng.random() < prob:
                self.seat_player(seat, str(self.rng.choice(kinds, p=weights / weights.sum())))

    def fill_initial_seats(self) -> list[SeatEvent]:
        self._arrivals(self.config.initial_occupancy)
        return self._drain()

    def expel(self, seat: int) -> SeatedPlayer | None:
        sp = self.seats[seat]
        if sp is not None:
            self._pending.append(self._event(seat, sp, "expelled"))
            self.seats[seat] = None
        return sp

    def _drain(self) -> list[SeatEvent]:
        events, self._pending = self._pending, []
        return events

    def _decide(self, seat: int, hand: Hand, up: int, allowed: frozenset[Action]) -> Action:
        sp = self.seats[seat]
        if sp.available < hand.bet:     # no le alcanza para doblar ni dividir
            allowed = allowed - {Action.DOUBLE, Action.SPLIT}
            if allowed == {Action.STAND}:
                return Action.STAND

        action = sp.player.decide(hand, up, allowed)

        flags = dict(
            can_double=Action.DOUBLE in allowed,
            can_split=Action.SPLIT in allowed,
            can_surrender=Action.SURRENDER in allowed,
        )
        basic = basic_strategy(hand, up, self.config.rules, **flags)
        if action != basic:
            sp.deviated = True
            index_action, play = counter_strategy(hand, up, self.tc, self.config.rules, **flags)
            sp.matched_index |= play is not None and action == index_action

        if action in (Action.DOUBLE, Action.SPLIT):
            sp.staked += hand.bet
        return action

    def _insure(self, seat: int, hand: Hand, up: int) -> bool:
        sp = self.seats[seat]
        if sp.available < hand.bet / 2 or not sp.player.insure(hand, up):
            return False
        sp.staked += hand.bet / 2
        # El seguro nunca es estrategia básica; sólo coincide con el índice si TC >= +3.
        sp.deviated = True
        sp.matched_index |= take_insurance(self.tc)
        return True

    def _on_card(self, card: int) -> None:
        self.running_count += hi_lo_value(card)
        for sp in self.seats:
            if sp is not None:
                sp.player.observe_card(card, self.shoe.decks_remaining)

    def _shuffle_if_needed(self) -> None:
        if self.shoe.needs_shuffle:
            self.shoe.shuffle()
            self.running_count = 0
            for sp in self.seats:
                if sp is not None:
                    sp.player.on_shuffle(self.shoe.decks_remaining)

    def _place_bets(self) -> dict[int, float]:
        bets = {}
        for seat, sp in enumerate(self.seats):
            if sp is None:
                continue
            bet = min(sp.player.place_bet(), self.config.max_bet_units, sp.bankroll)
            sp.staked = bet
            sp.deviated = sp.matched_index = False
            bets[seat] = bet
        return bets

    def play_round(self) -> list[SeatEvent]:
        """Juega una ronda y devuelve todos los eventos ocurridos desde la anterior.

        Si nadie apuesta, no se reparte (ni avanza el zapato), pero las sesiones sí
        corren: quien espera un buen conteo sin cartas en la mesa termina yéndose.
        """
        self.round_id += 1
        self._shuffle_if_needed()
        self._arrivals(self.config.arrival_rate)

        tc, penetration = self.tc, self.shoe.penetration_used
        bets = self._place_bets()
        results = {}
        if any(b > 0 for b in bets.values()):
            outcome = play_round(
                self.shoe, self.config.rules, bets, self._decide, self._insure, self._on_card
            )
            results = {r.seat: r for r in outcome.seats}

        for seat, sp in enumerate(self.seats):
            if sp is None:
                continue
            r = results.get(seat)
            actions: list[str] = []
            net = 0.0
            if r is not None:
                net = r.net
                actions = [a.value for a in r.actions]
                if r.insurance_bet > 0:
                    actions.insert(0, "insurance")
                sp.bankroll += net
                sp.player.on_round_end(net)
            sp.rounds_seen += 1
            sp.rounds_left -= 1
            self._pending.append(
                self._event(
                    seat,
                    sp,
                    "play",
                    bet=bets[seat],
                    true_count=tc,
                    shoe_penetration=penetration,
                    actions=actions,
                    deviated_from_basic=sp.deviated,
                    matched_index_play=sp.matched_index,
                    result_units=net,
                )
            )
            if sp.rounds_left <= 0 or sp.bankroll < 1.0:
                self._pending.append(self._event(seat, sp, "leave"))
                self.seats[seat] = None

        return self._drain()
