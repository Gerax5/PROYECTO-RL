# Supervisor de Mesa de Blackjack — Configuración del Proyecto (RL)

> Objetivo: entrenar un **agente supervisor** que observe una mesa de blackjack de 5 sillas, detecte contadores de cartas (abiertos y disimulados) y los expulse, **sin expulsar a jugadores casuales**, y que se **reentrene continuamente** con lo que observa.

---

## 1. Visión general de la arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                     CasinoTableEnv (Gymnasium)               │
│                                                              │
│  ┌────────┐   ┌───────────────────────────────────────────┐  │
│  │ Shoe   │──▶│ Mesa (5 sillas) + Dealer                  │  │
│  │ 6 mazos│   │  silla 1..5 ← Jugadores (casual/contador/ │  │
│  └────────┘   │                disimulado) entran/salen   │  │
│               └───────────────────────────────────────────┘  │
│                        │ eventos por ronda                   │
│                        ▼                                     │
│               ┌──────────────────┐                           │
│               │ Feature Extractor│ (historial por silla)     │
│               └──────────────────┘                           │
└────────────────────────│─────────────────────────────────────┘
                         ▼ observación
                ┌──────────────────┐   acción (mantener/expulsar por silla)
                │ Supervisor (RL)  │──────────────────────────────▶ Env
                └──────────────────┘
                         ▲
                         │ trayectorias
                ┌──────────────────┐
                │ Replay / Buffer  │ ← reentrenamiento continuo
                └──────────────────┘
```

- **Los jugadores NO aprenden en la mesa** (son políticas fijas/parametrizadas), salvo en la fase adversarial (sección 7.3).
- **El único agente que aprende es el supervisor.**

---

## 2. Stack tecnológico

| Componente | Herramienta |
|---|---|
| Lenguaje | Python 3.11 |
| Ambiente | `gymnasium` (≥ 0.29) |
| RL | `stable-baselines3` + `sb3-contrib` (`RecurrentPPO`) o `torch` propio |
| Redes | `torch` (≥ 2.x) |
| Configuración | `hydra-core` / archivos YAML |
| Logging | `tensorboard` (o `wandb`) |
| Datos / análisis | `numpy`, `pandas`, `matplotlib` |
| Tests | `pytest` |

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install gymnasium stable-baselines3 sb3-contrib torch numpy pandas matplotlib hydra-core tensorboard pytest
```

---

## 3. Estructura de carpetas sugerida

```
PROYECTO/
├── configs/
│   ├── env.yaml              # mesa, shoe, reglas, llegada de jugadores
│   ├── players.yaml          # parámetros de cada tipo de jugador
│   ├── supervisor.yaml       # red, algoritmo, hiperparámetros
│   └── training.yaml         # fases, curriculum, reentrenamiento
├── src/
│   ├── blackjack/            # ✅ implementado
│   │   ├── cards.py          # cartas y conteo Hi-Lo
│   │   ├── shoe.py           # baraja, corte, reshuffle
│   │   ├── rules.py          # reglas (S17/H17, DAS, splits, BJ 3:2) y acciones
│   │   ├── hand.py
│   │   ├── basic_strategy.py # tablas de estrategia básica
│   │   ├── deviations.py     # Illustrious 18 + seguro
│   │   └── engine.py         # motor de una ronda multi-silla (callbacks de decisión)
│   ├── players/
│   │   ├── base_player.py
│   │   ├── casual.py
│   │   ├── counter.py
│   │   └── stealth_counter.py
│   ├── env/
│   │   ├── table.py          # 5 sillas, rondas, entrada/salida
│   │   ├── features.py       # extracción de features por silla
│   │   └── casino_env.py     # wrapper Gymnasium
│   ├── supervisor/
│   │   ├── policy.py         # red (GRU/Transformer por silla)
│   │   └── train.py
│   ├── retraining/
│   │   ├── buffer.py
│   │   ├── online_loop.py
│   │   └── adversary.py      # evolución de contadores disimulados
│   └── eval/
│       ├── metrics.py
│       └── evaluate.py
├── tests/
├── notebooks/
└── CONFIGURACION_PROYECTO.md
```

---

## 4. Configuración de la mesa y el juego (`configs/env.yaml`)

```yaml
table:
  seats: 5
  min_bet: 10
  max_bet: 500
  rounds_per_episode: 600        # un "turno" del supervisor

shoe:
  n_decks: 6
  penetration: 0.75              # se baraja al 75% del shoe
  burn_cards: 1

rules:
  dealer_hits_soft_17: false     # S17
  blackjack_payout: 1.5          # 3:2
  double_after_split: true
  max_splits: 3
  surrender: false
  insurance: true                # importante: los contadores la toman con TC ≥ +3

arrivals:                        # dinámica de entrada/salida
  arrival_rate: 0.15             # prob. de que llegue alguien a una silla vacía por ronda
  population_mix:                # probabilidad del tipo del que llega
    casual: 0.70
    counter: 0.15
    stealth_counter: 0.15
  session_length_rounds:         # cuánto se queda en promedio (distribución)
    casual:          {dist: geometric, mean: 80}
    counter:         {dist: geometric, mean: 120}
    stealth_counter: {dist: geometric, mean: 100}
  bankroll_units: [50, 400]      # bankroll inicial en unidades (uniforme)

seed: 42
```

**Información pública** (la que ve el supervisor, igual que un pit boss real): cartas repartidas, apuestas, decisiones de cada jugador, resultados, entradas/salidas, penetración del shoe.
**Información privada** (NO la ve): el tipo real de jugador, su conteo interno, su bankroll.

---

## 5. Modelos de jugadores (`configs/players.yaml`)

Todos los jugadores juegan blackjack con una política base; lo que los diferencia es **cómo apuestan** y **cuándo se desvían de la estrategia básica**.

### 5.1 Casual
- Estrategia básica con **tasa de error** (p.ej. 5–20% de decisiones subóptimas).
- Apuesta basada en sesgos humanos, **no en el conteo**:
  - "Racha caliente": sube tras ganar.
  - Martingala parcial: dobla tras perder.
  - Apuesta plana con ruido.
- Llega y se va de forma independiente al conteo.

### 5.2 Contador (abierto)
- Conteo **Hi-Lo**; `TC = RC / mazos_restantes`.
- Bet spread agresivo con rampa 1-2-4-8-12: 1 unidad con TC ≤ 1, 2 con TC 2, 4 con TC 3, 8 con TC 4, 12 con TC ≥ 5.
  (Una rampa lineal `TC − 1` es demasiado suave: en simulación da sólo +0.19 u/100 rondas contra +1.23 u/100 de esta rampa.)
- Aplica las desviaciones **Illustrious 18** (p.ej. plantarse 16 vs 10 con TC ≥ 0, seguro con TC ≥ +3).
- Puede hacer **wonging**: entra con TC alto, se va cuando el TC cae.

### 5.3 Contador disimulado
Mismo conteo, pero con **camuflaje parametrizable** (estos parámetros son la "dificultad"):

```yaml
stealth_counter:
  counting_system: hi_lo
  bet_spread_max: 4              # spread pequeño (1–4 en vez de 1–12)
  bet_noise_std: 0.3             # ruido en la apuesta
  cover_bet_prob: 0.25           # apuesta "como casual" tras ganar/perder
  index_play_prob: 0.5           # sólo aplica la mitad de las desviaciones
  deliberate_error_prob: 0.03    # errores intencionales de estrategia
  wonging: false
  delay_bet_ramp_rounds: 1       # sube la apuesta con retraso para romper la correlación
  disguise_level: 0.0 - 1.0      # escalar que interpola entre "contador" y "casual"
```

> `disguise_level` permite hacer **curriculum** (sección 7.2): 0 = idéntico al contador abierto, 1 = casi indistinguible.

> **Medido (4M de rondas, IC95):** con estos parámetros el disimulado gana con nivel 0.3 (+0.37 u/100, [+0.11, +0.63]) y **pierde** desde 0.6 (−0.27 u/100, [−0.50, −0.04]; nivel 1.0: −0.97 u/100). Por eso `TableConfig` usa 0.3 por defecto. Los niveles altos del curriculum (0.6 y 0.9) necesitan otro camuflaje, o el supervisor recibirá premio por expulsar a jugadores que le dan ganancia a la casa.

---

## 6. El supervisor (agente RL)

### 6.1 Formulación
Es un **POMDP**: el supervisor no ve el tipo del jugador, sólo su comportamiento. Por eso la política necesita **memoria** (RNN/Transformer) o features agregadas sobre una ventana.

### 6.2 Espacio de observación
Para cada una de las 5 sillas, un vector de features calculado sobre el historial de ese jugador (ventana deslizante, p.ej. últimas 50 rondas) + features globales.

**Features por silla** (`src/env/features.py`):

| Feature | Qué captura |
|---|---|
| `occupied` | silla ocupada (0/1) |
| `rounds_seen` | cuántas rondas lleva (confianza de la estimación) |
| `bet_norm` | apuesta actual / apuesta mínima del jugador |
| `corr_bet_tc` | correlación de Pearson/Spearman entre apuesta y TC **calculado por el supervisor** ← señal clave |
| `bet_spread` | max/min de la apuesta en la ventana |
| `bet_tc_slope` | pendiente de regresión apuesta ~ TC |
| `corr_bet_prev_result` | correlación con resultado previo (típico de casual) |
| `index_play_rate` | % de desviaciones que coinciden con I18 cuando el TC lo justifica |
| `basic_strategy_error_rate` | % de errores vs. estrategia básica |
| `insurance_tc_mean` | TC promedio cuando toma seguro |
| `entry_tc` / `exit_tc` | TC al entrar/salir (detecta wonging) |
| `net_win_units` | ganancia acumulada del jugador |

**Features globales**: TC actual, penetración del shoe, número de jugadores, número de expulsiones en el turno.

```python
observation_space = gym.spaces.Dict({
    "seats":  gym.spaces.Box(-np.inf, np.inf, shape=(5, N_FEATURES), dtype=np.float32),
    "global": gym.spaces.Box(-np.inf, np.inf, shape=(N_GLOBAL,),    dtype=np.float32),
})
```

> Variante "end-to-end": en vez de features hechas a mano, pasar la secuencia cruda de eventos por silla (apuesta, TC, acción, resultado) a una GRU/Transformer. Recomendado como segunda iteración, para comparar.

### 6.3 Espacio de acción
Por ronda, una decisión por silla:

```python
action_space = gym.spaces.MultiDiscrete([3] * 5)
# 0 = no hacer nada
# 1 = vigilar (observación detallada: features más precisas, coste pequeño)
# 2 = expulsar
```

Versión simplificada para empezar: `Discrete(6)` → expulsar silla `i` (0–4) o no hacer nada (5).

### 6.4 Función de recompensa

```yaml
reward:
  expel_counter:          +10.0
  expel_stealth_counter:  +15.0     # vale más detectar al disimulado
  expel_casual:           -25.0     # falso positivo: muy penalizado (cliente perdido)
  counter_edge_per_round: -1.0      # × ganancia esperada del contador en esa ronda (pérdida de la casa)
  watch_cost:             -0.05     # coste de "vigilar"
  early_detection_bonus:  +k / rounds_seen   # premia detectar rápido
```

Idea central: **expulsar tarde cuesta dinero, expulsar mal cuesta clientes**. El balance `expel_casual` vs. `expel_counter` define la tolerancia a falsos positivos; ajustarlo y reportar la curva.

### 6.5 Arquitectura de la política (`configs/supervisor.yaml`)

Encoder **compartido por silla** (mismos pesos para las 5 sillas → invariante a la posición) + contexto global:

```
features silla_i ──▶ MLP compartido ──▶ GRU (memoria) ──▶ h_i ─┐
                                                               ├─▶ cabeza de acción por silla (3 logits)
features globales ──▶ MLP ──────────────────────────────────▶ g ─┘
                                                            └─▶ cabeza de valor V(s)
```

```yaml
algorithm: RecurrentPPO
policy:
  seat_encoder: [64, 64]
  rnn: {type: gru, hidden: 128}
  global_encoder: [32]
ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 256
  gamma: 0.995                   # horizonte largo: la detección tarda muchas rondas
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  n_envs: 8                      # mesas en paralelo
total_timesteps: 5_000_000
```

Alternativas a comparar: DQN con features agregadas (baseline), PPO sin memoria, clasificador supervisado + umbral (baseline no-RL).

---

## 7. Entrenamiento y reentrenamiento continuo (`configs/training.yaml`)

### 7.1 Fase 0 — Baselines
1. Regla heurística: expulsar si `corr_bet_tc > 0.6` y `rounds_seen > 30`.
2. Clasificador supervisado (logistic regression / gradient boosting) sobre las features, usando las etiquetas reales del simulador.

Sirven para saber si el RL aporta algo.

### 7.2 Fase 1 — Curriculum
Subir progresivamente la dificultad de los disimulados:

```yaml
curriculum:
  stages:
    - {disguise_level: 0.0, stealth_share: 0.05, until: {detection_rate: 0.90}}
    - {disguise_level: 0.3, stealth_share: 0.10, until: {detection_rate: 0.85}}
    - {disguise_level: 0.6, stealth_share: 0.15, until: {detection_rate: 0.80}}
    - {disguise_level: 0.9, stealth_share: 0.15, until: {timesteps: 2_000_000}}
  max_false_positive_rate: 0.05   # no avanzar si el supervisor expulsa casuales de más
```

### 7.3 Fase 2 — Reentrenamiento continuo ("con lo que va viendo")
Bucle online (`src/retraining/online_loop.py`):

```
loop:
  1. Desplegar la política actual en K mesas durante N turnos.
  2. Guardar trayectorias + etiquetas reales (el simulador conoce el tipo).
  3. Añadir al buffer (mezcla: 70% datos nuevos, 30% históricos → evita olvido catastrófico).
  4. Fine-tuning de PPO por M iteraciones.
  5. Evaluar en un set de validación FIJO (mismas semillas).
  6. Si mejora → promover como nueva versión (checkpoint v_{t+1}); si no → descartar.
```

**Adversario evolutivo (arms race)** — `src/retraining/adversary.py`:
- Los parámetros de camuflaje del contador disimulado se optimizan (algoritmo evolutivo / CMA-ES, o RL) para **maximizar su ganancia esperada − penalización por ser expulsado** contra el supervisor actual.
- El supervisor se reentrena contra la nueva población de disimulados.
- Mantener una **liga** de adversarios de versiones anteriores para no "olvidar" cómo detectarlos.

```yaml
retraining:
  shifts_per_cycle: 200
  buffer_size: 500_000
  replay_ratio_old: 0.3
  finetune_iterations: 50
  promote_if: {metric: f1_counters, min_improvement: 0.01, max_fp_rate: 0.05}
adversary:
  method: cma_es
  population: 32
  generations_per_cycle: 10
  fitness: ev_per_round - 50 * expelled
  league_size: 10
```

> Nota de realismo: en un casino real no hay etiquetas inmediatas. Para simularlo, opción de **etiquetas retardadas/ruidosas** (p.ej. sólo se confirma el tipo con prob. 0.5 y con retraso de varios turnos).

---

## 8. Métricas de evaluación (`src/eval/metrics.py`)

| Métrica | Descripción |
|---|---|
| Recall contadores | % de contadores abiertos expulsados |
| Recall disimulados | % de disimulados expulsados ← métrica principal |
| Tasa de falsos positivos | % de casuales expulsados (meta < 5%) |
| Precision / F1 | sobre todas las expulsiones |
| Tiempo a detección | rondas desde que se sienta hasta que es expulsado |
| Edge de la casa | ganancia de la casa por ronda (con vs. sin supervisor) |
| Pérdida evitada | dinero que los contadores dejaron de ganar |
| Curva ROC / trade-off | variando `expel_casual` |
| Robustez | rendimiento vs. `disguise_level` (0 → 1) |

Graficar todo por **ciclo de reentrenamiento** para mostrar que el supervisor mejora con el tiempo.

---

## 9. Checklist de implementación

`[A]` / `[B]` = responsable (ver sección 11).

- [ ] `[A+B]` Contrato de interfaces (`EventLog`, espacios de observación/acción) — sección 11.2.
- [x] `[A]` Shoe, reglas y mano (`src/blackjack/`) + tests de pago y reglas.
- [x] `[A]` Tabla de estrategia básica + desviaciones I18.
- [x] `[A]` Jugadores: casual, contador, disimulado (con `disguise_level`).
- [x] `[A]` Verificar por simulación que el contador tiene EV positivo y el casual negativo (`scripts/validate_players.py`).
- [ ] `[A]` Conseguir EV > 0 en el disimulado con `disguise_level` ≥ 0.6 (hoy pierde dinero; ver sección 5.3).
- [x] `[A]` Mesa de 5 sillas con entrada/salida de jugadores.
- [ ] `[B]` Extractor de features por silla.
- [x] `[A]` `CasinoTableEnv` (Gymnasium) + `check_env`.
- [ ] `[B]` Baselines (heurística y clasificador).
- [ ] `[B]` Supervisor RecurrentPPO + logging en TensorBoard.
- [ ] `[A]` Curriculum de disfraz.
- [ ] `[B]` Bucle de reentrenamiento continuo + buffer.
- [ ] `[A]` Adversario evolutivo + liga.
- [ ] `[B]` Métricas y script de evaluación.
- [ ] `[A+B]` Experimentos finales, gráficas e informe.

---

## 10. Riesgos y decisiones abiertas

- **Recompensa desbalanceada**: si `expel_casual` es poco penalizado, el agente expulsa a todos. Monitorear siempre la tasa de FP.
- **Pocas señales en pocas rondas**: con < 20 rondas casi no hay información; la red debe aprender a "esperar".
- **Reward hacking**: verificar que el supervisor no explote artefactos del simulador (p.ej. la duración de sesión distinta por tipo). Usar la misma distribución de duración para todos en evaluación.
- **Sobreajuste al adversario**: la liga de adversarios antiguos y el set de validación fijo lo mitigan.
- **Varianza alta del blackjack**: usar muchas mesas en paralelo y semillas fijas para comparar versiones.

---

## 11. División del trabajo (2 personas)

El proyecto se divide en dos mitades con un **contrato de interfaces** entre ellas, para que cada persona avance sin bloquear a la otra.

| | **Persona A — Simulación y adversarios** | **Persona B — Supervisor y aprendizaje** |
|---|---|---|
| Pregunta que responde | ¿Cómo se comporta la mesa y cada jugador? | ¿Cómo detecta y aprende el supervisor? |
| Carpetas | `src/blackjack/`, `src/players/`, `src/env/table.py`, `src/env/casino_env.py`, `src/retraining/adversary.py` | `src/env/features.py`, `src/supervisor/`, `src/retraining/buffer.py`, `src/retraining/online_loop.py`, `src/eval/` |
| Configs | `env.yaml`, `players.yaml`, sección `curriculum` y `adversary` de `training.yaml` | `supervisor.yaml`, sección `reward` y `retraining` de `training.yaml` |

### 11.1 Tareas por persona

**Persona A — Simulación y adversarios**
1. Motor de blackjack: shoe (6 mazos, penetración, reshuffle), reglas, manos, pagos + tests.
2. Estrategia básica + desviaciones Illustrious 18.
3. Jugador casual (sesgos de apuesta, tasa de error).
4. Contador abierto (Hi-Lo, spread 1–12, I18, wonging).
5. Contador disimulado con `disguise_level` y parámetros de camuflaje.
6. Validación: simular ~1M de manos y confirmar EV (contador > 0, casual < 0, disimulado > 0 pero menor).
7. Mesa de 5 sillas: llegadas, salidas, bankroll, expulsiones.
8. `CasinoTableEnv` (Gymnasium): `reset`, `step`, aplicar acciones, calcular recompensa con los pesos de `reward` + `check_env`.
9. Curriculum: generador de poblaciones por etapa de `disguise_level`.
10. Adversario evolutivo (CMA-ES) + liga de adversarios.

**Persona B — Supervisor y aprendizaje**
1. Extractor de features por silla (correlación apuesta–TC, spread, tasa de I18, wonging, etc.).
2. Generador de datos *mock* (logs falsos con el formato del contrato) para empezar sin esperar a A.
3. Baselines: regla heurística + clasificador supervisado (logistic regression / gradient boosting).
4. Arquitectura de la política (encoder compartido por silla + GRU + cabezas).
5. Entrenamiento con RecurrentPPO, `SubprocVecEnv` con 8 mesas, TensorBoard.
6. Diseño y ajuste de la recompensa (barrido de `expel_casual` → curva de trade-off).
7. Buffer de experiencia y bucle de reentrenamiento online (mezcla 70/30, promoción de checkpoints).
8. Métricas (recall por tipo, FP, tiempo a detección, edge de la casa) y script de evaluación con semillas fijas.

**Ambos**
- Definir el contrato de interfaces (semana 1).
- Experimentos finales, gráficas, informe y presentación (cada uno escribe la sección de su parte).
- Revisión cruzada: cada uno revisa el código del otro antes de integrarlo.

### 11.2 Contrato de interfaces (se fija en la semana 1)

```python
# Evento que la mesa (A) emite por silla y por ronda; lo consume el extractor de features (B)
@dataclass
class SeatEvent:
    round_id: int
    seat: int                 # 0..4
    player_id: int | None     # None si la silla está vacía
    event: str                # "sit" | "play" | "leave" | "expelled"
    bet: float
    true_count: float         # TC público al inicio de la ronda (calculable por el supervisor)
    actions: list[str]        # ["hit", "stand", "double", "split", "insurance", ...]
    deviated_from_basic: bool
    matched_index_play: bool
    result_units: float
    shoe_penetration: float
    true_type: str            # "casual" | "counter" | "stealth" — SOLO para métricas/etiquetas, nunca en la observación
```

- **A entrega**: `CasinoTableEnv` que devuelve `info["events"]: list[SeatEvent]` y aplica `MultiDiscrete([3]*5)`.
- **B entrega**: `FeatureExtractor.transform(events) -> obs` con la forma definida en la sección 6.2, y la usa A dentro del env para construir la observación.
- La recompensa la **calcula el env (A)** con los pesos que **define B** en `training.yaml`.

### 11.3 Cronograma sugerido (8 semanas)

| Semana | Persona A | Persona B | Punto de sincronización |
|---|---|---|---|
| 1 | Shoe, reglas, manos + tests | Datos mock + extractor de features | ✅ Contrato de interfaces firmado |
| 2 | Estrategia básica, I18, casual | Baselines sobre datos mock | |
| 3 | Contador y disimulado + validación de EV | Arquitectura de la política | ✅ Jugadores validados (EV) |
| 4 | Mesa 5 sillas + `CasinoTableEnv` | Integrar features en el env, primer entrenamiento PPO | ✅ **Integración**: env + supervisor corriendo end-to-end |
| 5 | Curriculum de disfraz | Ajuste de recompensa + baselines sobre el env real | |
| 6 | Adversario evolutivo (CMA-ES) | Buffer + bucle de reentrenamiento online | ✅ Curriculum funcionando |
| 7 | Liga de adversarios + integración con el bucle | Métricas, evaluación y gráficas | ✅ Arms race A↔B funcionando |
| 8 | Experimentos e informe (parte de simulación) | Experimentos e informe (parte de aprendizaje) | ✅ Entrega |

### 11.4 Por qué la división es equitativa
- **A** tiene más ingeniería de simulación (motor, jugadores, env) y la parte de optimización adversarial.
- **B** tiene más trabajo de ML/experimentación (features, red, PPO, reentrenamiento, evaluación), que requiere muchas iteraciones de entrenamiento.
- Ambos tocan RL: A con el diseño del POMDP y el adversario, B con el agente y su entrenamiento.
- Gracias a los datos *mock*, B no depende de A hasta la semana 4.
