# Botgame — contexto de desarrollo

## Qué es este proyecto

Bot en Python que corre en el **PC** y maneja un móvil Android **físico por USB** (también BlueStacks) vía ADB. El objetivo es jugar automáticamente a videojuegos propios usando imitation learning (aprender de un vídeo de gameplay), y usarlo para **red-teamear el sistema anti-bot** del propio juego: se mide qué detecta y qué no.

**No** está destinado a evadir anti-cheat de terceros.

---

## Estado actual: 9 bloques completados

### Bloque 1 — Bridge ADB
- `src/botgame/adb/device.py` — autodetecta el móvil USB, lee tamaño de pantalla
- `src/botgame/adb/capture.py` — screenshot via `adb exec-out screencap -p`
- `src/botgame/adb/input.py` — tap / swipe / swipe_path (input adb)
- `src/botgame/bots/dummy.py` — bot tonto que ejercita el pipeline completo

### Bloque 2 — Vídeo → Dataset
- `src/botgame/dataset/getevent.py` — parsea `adb shell getevent -lt` en tap/swipe con timestamps
- `src/botgame/dataset/touch_overlay.py` — detecta overlay "Mostrar toques" en frames (numpy)
- `src/botgame/dataset/video.py` — extrae frames de un vídeo con OpenCV (import lazy)
- `src/botgame/dataset/builder.py` — alinea frames con acciones → `frames/*.npy` + `labels.jsonl`
- CLI: `python -m botgame build-dataset --video play.mp4 --events events.log --fps 10 --out dataset`

### Bloque 3 — Modelo + Policy Bot
- `src/botgame/model/encoding.py` — codificación acción ↔ target (type index + coords normalizadas)
- `src/botgame/model/net.py` — CNN pequeña (3 Conv + AvgPool + 2 FC) → type logits + coords
- `src/botgame/model/dataset.py` — torch Dataset sobre el directorio del Bloque 2
- `src/botgame/model/train.py` — entrenamiento por behavioral cloning (CE type + MSE coords maskeada)
- `src/botgame/model/infer.py` — clase `Policy` que carga checkpoint y predice Action desde frame numpy
- `src/botgame/bots/policy.py` — `PolicyBot`: captura → modelo → humanización → toque → telemetría
- CLI: `python -m botgame train --dataset dataset --screen-size 1080 2400 --epochs 20 --out policy.pt`
- CLI: `python -m botgame run-policy --model policy.pt --level 0.5`

### Bloque 4 — Captura rápida (streaming H.264)
- `src/botgame/adb/fast_capture.py` — `FastCapture`: `adb exec-out screenrecord --output-format=h264 -` →
  stdout streaming → PyAV decode en hilo background → slot `_latest` (mutex). `grab()` devuelve último frame.
  Auto-restart antes del cap de 180 s de `screenrecord`.
- `AdbDevice.popen_exec_out(...)` — helper para spawn de `adb exec-out` como `Popen` (lectura streaming).
- `DummyBot` / `PolicyBot` aceptan `capture=` inyectable (duck-typed `.grab()`) → drop-in fast/slow.
- CLI flag `--fast` en `run-dummy` y `run-policy`. ExitStack maneja lifecycle `start()/stop()`.
- Dep opcional: `pip install av` (lazy import; error claro si falta).
- Salto: ~1-5 fps (screencap PNG) → ~30 fps (H.264 stream).

### Bloque 5 — Input rápido (minitouch)
- `src/botgame/adb/minitouch.py` — `MiniTouch` (drop-in de `TouchInput`):
  `adb forward tcp:<port> localabstract:minitouch` → socket TCP → parsea banner
  (`v`, `^ max_contacts max_x max_y max_pressure`, `$ pid`) → reescala coords screen-px a touch-panel.
  Protocolo: `d/m/u/c/w`. `swipe_path` envía gesto continuo (no segmentos como `input swipe`).
  `key()` falla a `adb shell input keyevent` (minitouch no maneja keyevents).
- Bots aceptan `touch=` inyectable; CLI flags `--minitouch` + `--minitouch-port`.
- Prereq manual: push del binario + start del daemon (documentado en README).

### Bloque 6 — Loop red-team automatizado
- `src/botgame/redteam/sweep.py` — `run_sweep(bot_factory, detector, levels, ...)`:
  - `bot_factory(level, seed, path) → path` corre 1 sesión, escribe telemetría JSONL
  - `detector(path) → float ∈ [0,1]` clasifica sesión (1 = bot probable)
  - Valida levels upfront (fail-fast antes de ejecutar nada)
  - Escribe CSV `level,seed,telemetry,score`
- `sample_detector` — heurística demo (ratio reaction_s=0 + jitter=0). No para producción.
- `load_detector("mod:func")` — resuelve detector via import spec.
- CLI: `python -m botgame redteam --levels 0 0.25 0.5 0.75 1 --steps 200 --detector pkg:fn`
- `--dry-run` sintetiza telemetría desde el Humanizer sin dispositivo (smoke / CI).

### Bloque 7 — PPO fine-tuning sobre BC
- `src/botgame/rl/env.py` — `BotEnv(capture, touch, reward_fn, ...)` y `RandomEnv` (stub CPU).
  Interfaz mínima Gym: `reset() → frame`, `step(action) → EnvStep(frame, reward, done, info)`.
- `src/botgame/rl/net.py` — `ActorCritic`: mismo trunk que `PolicyNet` + `value_head` + `log_sigma` aprendido.
  `load_bc_weights(ac, ckpt)` copia pesos compatibles del checkpoint BC (skip value_head + log_sigma).
- `src/botgame/rl/ppo.py` — PPO mínimo PyTorch:
  - Sampleo: type ~ Categorical, coords ~ Normal(mean, exp(log_sigma)) clip [0,1]
  - GAE + ratio clip + value MSE + entropy bonus
  - `train(total_steps) → list[RolloutStats]`, `save(path)`
- CLI: `python -m botgame train-rl --env random --bc policy.pt --out policy_rl.pt --steps 5000`
- `BotEnv` para uso real solo via Python (reward_fn es game-specific).

### Bloque 8 — Librería detectores + multi-sesión + plot
- `src/botgame/redteam/detectors.py` — 4 primitivas + composite:
  - `periodicity_detector` — CV(intervalos). Metronómico = 1.
  - `coord_cluster_detector` — std-dev coords. Mismo píxel = 1.
  - `perfect_aim_detector` — fracción target==actual.
  - `reaction_time_detector` — fracción reaction_s < 80ms (sub-humano).
  - `composite_detector(dets, weights)` + `default_composite` (equal-weight de las 4).
- `run_sweep` ahora soporta `sessions_per_level=N` (seeds únicos por sesión).
- `aggregate_by_level(results)` → mean/std/n por level.
- `src/botgame/redteam/plot.py` — `plot_sweep(csv, png)` con errorbars (matplotlib lazy import).
- CLI: `--sessions-per-level`, `--plot` en `redteam`; nuevo subcomando `redteam-plot`.
- Curva demo con `default_composite` + dry-run: 0.75 (level 0) → 0.08 (level 1).

### Bloque 9 — Primitivas reward RL
- `src/botgame/rl/rewards.py`:
  - `pixel_diff_reward(scale, downsample)` — proxy "algo cambió".
  - `region_brightness_reward(x,y,w,h,baseline)` — score-HUD lit up.
  - `template_match_reward(template, threshold, polarity)` — NCC pura-numpy (test-friendly).
  - `compose_rewards([(fn, weight), ...])` — combinación lineal.
- Re-exportadas en `botgame.rl.__all__` para `from botgame.rl import compose_rewards, ...`.

### Humanización (clave para red-team)
- `src/botgame/humanize.py` — perilla `level` 0→1: dispersión de toque, latencia de reacción, jitter de intervalo, trayectoria Bézier con temblor. Seedable para reproducibilidad.

### Telemetría
- `src/botgame/telemetry.py` — JSONL append-only con timestamp, acción, coords reales vs objetivo, nivel de humanización.

---

## Tests
```
80 passed  (humanize, dataset, model, fast_capture, minitouch, redteam,
            detectors, redteam_plot, rl, rewards)
```
Ejecutar: `PYTHONPATH=src python3 -m pytest -q`

---

## Comandos de uso rápido

```bash
# 1. Conectar móvil y verificar
adb devices
python -m botgame devices

# 2. Screenshot de prueba
python -m botgame cap --out screen.png

# 3. Grabar gameplay + eventos táctiles simultáneamente
#    (en dos shells)
adb shell getevent -lt > events.log      # shell 1: captura toques
# [graba el gameplay con el móvil en mano]  shell 2: graba pantalla

# 4. Construir dataset
python -m botgame build-dataset \
    --video play.mp4 --events events.log \
    --src-size 1080 2400 --dst-size 1080 2400 \
    --fps 10 --out dataset

# 5. Entrenar
python -m botgame train \
    --dataset dataset --screen-size 1080 2400 \
    --epochs 20 --out policy.pt

# 6. Jugar live (sweep de level para testear el detector)
python -m botgame run-policy --model policy.pt --level 0.0
python -m botgame run-policy --model policy.pt --level 1.0
```

---

## Dependencias

```
pip install numpy Pillow torch
pip install opencv-python   # solo para build-dataset
pip install av              # solo para FastCapture (--fast)
pip install matplotlib      # solo para redteam --plot / redteam-plot
```

---

## Siguientes pasos pendientes

- Conectar un dispositivo real y validar end-to-end: `cap`, `run-dummy --fast --minitouch`, `run-policy`.
- Entrenar un BC sobre tu propio gameplay (`build-dataset` + `train`) y `train-rl --bc ...` con un `reward_fn` específico del juego.
- Sustituir `sample_detector` por tu detector real (`--detector pkg.module:func`) y trazar la curva detector vs `level`.

---

## Arquitectura resumida

```
[móvil USB]
     |
     | adb (screencap / input)
     |
[PC]
  capture → frame (H×W×3 numpy)        ← screencap PNG (lento) | screenrecord H.264 + PyAV (fast)
     ↓
  policy.predict(frame) → Action      ← modelo CNN (policy.pt)
     ↓
  humanizer.jitter / reaction_delay / bezier_path
     ↓
  touch.tap / swipe_path              → [móvil]
     ↓
  telemetry.log(...)                  → telemetry/*.jsonl
```

---

## Rama git

Rama activa: `Botgame`
Repositorio: `YiahMadafaka131/Botgame`

Para clonar y continuar en local:
```bash
git clone <repo-url>
cd Botgame
git checkout Botgame
pip install numpy Pillow torch opencv-python pytest
PYTHONPATH=src python3 -m pytest -q   # debe pasar 20 tests
```
