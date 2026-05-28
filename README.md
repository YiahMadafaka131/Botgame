# Botgame

PC-driven Android game-playing bot over ADB. Built for testing **your own**
games and **your own** bot-detection system: capture the screen, decide an
action, inject input, and log everything so detector verdicts can be
cross-referenced against ground-truth bot activity.

> Scope: intended for your own games / authorized red-teaming of your own
> detector. Not for evading third-party anti-cheat or automating games whose
> terms forbid it.

## Status

Block 1 of the roadmap: the **ADB bridge** (capture + input), a tunable
**humanization** layer, **telemetry**, and a **dummy bot** that exercises the
full pipeline. Perception and the learned policy come next.

## Requirements

- Python 3.10+
- Android **platform-tools** (`adb` on your PATH)
- A phone with **USB debugging** enabled, connected by cable
- `pip install -r requirements.txt`

## Connect your phone (USB)

1. On the phone: Settings → About → tap *Build number* 7× to unlock Developer
   options, then enable **USB debugging**.
2. Plug in by USB and accept the RSA fingerprint prompt.
3. Verify:
   ```bash
   adb devices            # your serial should show as "device"
   python -m botgame devices
   ```

## Usage

```bash
export PYTHONPATH=src      # or: pip install -e .

# Screenshot
python -m botgame cap --out captures/screen.png

# Single inputs
python -m botgame tap 540 1200
python -m botgame swipe 200 1000 800 1000 --duration 250

# Dummy bot (taps a region; --level sweeps robotic 0.0 -> human 1.0)
python -m botgame run-dummy --level 0.0 --steps 50            # robotic baseline
python -m botgame run-dummy --level 1.0 --swipes --seed 42    # human-like swipes
python -m botgame run-dummy --region 100 400 900 1600 --interval 0.6
```

Actions are logged to `telemetry/*.jsonl` (timestamp, target vs actual coords,
reaction delay, humanization level).

## Testing your detector (the red-team loop)

1. Run `run-dummy` across a sweep of `--level` (e.g. 0.0, 0.25, 0.5, 0.75, 1.0)
   with fixed `--seed` for reproducibility.
2. Have your detector label each session.
3. Plot detection rate vs. level → you get a curve showing which signals your
   detector relies on and where the gaps are.

## Architecture

```
capture (screencap)        -> botgame/adb/capture.py
  -> perception   [TODO]      (frame -> game state)
  -> policy       [TODO]      (state -> action; imitation learning from video)
  -> humanize                 botgame/humanize.py   (robotic <-> human knob)
  -> actuation (input)        botgame/adb/input.py
  -> telemetry                botgame/telemetry.py
```

## Roadmap

- [x] **Block 1** — ADB bridge + humanization + telemetry + dummy bot
- [ ] **Block 2** — video → (state, action) dataset for imitation learning
- [ ] **Block 3** — perception model + learned policy
- [ ] Faster capture (scrcpy/minicap) and minitouch backend for continuous gestures
```
