# Botgame — contexto de desarrollo

## Qué es este proyecto

Bot en Python que corre en el **PC** y maneja un móvil Android **físico por USB** (también BlueStacks) vía ADB. El objetivo es jugar automáticamente a videojuegos propios usando imitation learning (aprender de un vídeo de gameplay), y usarlo para **red-teamear el sistema anti-bot** del propio juego: se mide qué detecta y qué no.

**No** está destinado a evadir anti-cheat de terceros.

---

## Estado actual: 3 bloques completados

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

### Humanización (clave para red-team)
- `src/botgame/humanize.py` — perilla `level` 0→1: dispersión de toque, latencia de reacción, jitter de intervalo, trayectoria Bézier con temblor. Seedable para reproducibilidad.

### Telemetría
- `src/botgame/telemetry.py` — JSONL append-only con timestamp, acción, coords reales vs objetivo, nivel de humanización.

---

## Tests
```
20 passed  (test_humanize.py, test_dataset.py, test_model.py)
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
```

---

## Siguientes pasos pendientes (próximos bloques)

- **Captura rápida**: reemplazar `screencap` (1-5 fps) por stream **scrcpy** o **minicap** (30-60 fps). El bot real necesita reaccionar en tiempo real; con screencap es demasiado lento para juegos de acción.
- **Input de alta frecuencia**: reemplazar `adb shell input` por **minitouch** (gestos continuos, multitouch, sin latencia del proceso).
- **RL fine-tuning**: arrancar desde el checkpoint de imitation learning y afinar con Proximal Policy Optimization (PPO) usando la puntuación del juego como recompensa.
- **Loop red-team automatizado**: script que corre el bot en todo el espectro de humanización, llama al detector y devuelve la curva ROC — todo sin intervención manual.

---

## Arquitectura resumida

```
[móvil USB]
     |
     | adb (screencap / input)
     |
[PC]
  capture → frame (H×W×3 numpy)
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
