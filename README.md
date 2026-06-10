# Botgame

PC-driven Android game-playing bot over ADB. Built for testing **your own**
games and **your own** bot-detection system: capture the screen, decide an
action, inject input, and log everything so detector verdicts can be
cross-referenced against ground-truth bot activity.

> Scope: intended for your own games / authorized red-teaming of your own
> detector. Not for evading third-party anti-cheat or automating games whose
> terms forbid it.

## Status

- **Block 1** — the **ADB bridge** (capture + input), a tunable **humanization**
  layer, **telemetry**, and a **dummy bot** that exercises the full pipeline.
- **Block 2** — the **video → dataset** pipeline for imitation learning: turn a
  gameplay recording into labelled (frame, action) samples.
- **Block 3** — the **policy model**: a small CNN trained by behavioral cloning
  to predict an action (type + coordinates) from a frame, plus a `PolicyBot`
  that plays live through the ADB bridge.
- **Block 4** — the **replicator**: deterministic replay of the actions
  recovered from a gameplay video, with a **health monitor** that hunts for
  bugs (crashes, ANRs, lost focus, frozen screens) and saves evidence.

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

## Build a dataset from a gameplay video (Block 2)

Record yourself playing, then turn it into labelled (frame, action) samples for
imitation learning. Two ways to recover the actions:

**Preferred — getevent (pixel-accurate):** while recording, also capture touch
events. Then parse them, rescaling touch-device coords to screen pixels:

```bash
# during recording (separate shell): adb shell getevent -lt > events.log
python -m botgame build-dataset --video play.mp4 --events events.log \
    --src-size 1080 2400 --dst-size 1080 2400 --fps 10 --out dataset
```

**Fallback — show-touches overlay:** if you only have a screen recording, enable
Developer options → "Show taps" before recording and detect the overlay blob:

```bash
python -m botgame build-dataset --video play.mp4 --fps 10 --out dataset
```

Output: `dataset/frames/000000.npy …` (downscaled) + `dataset/labels.jsonl`
(one record per frame: frame path, action type, coords, timestamp).

## Train a policy and play (Block 3)

Train a behavioral-cloning model on your dataset, then let it play live:

```bash
# Train (screen-size = the px space the dataset coords are in)
python -m botgame train --dataset dataset --screen-size 1080 2400 \
    --epochs 20 --out policy.pt

# Play live through the ADB bridge (sweep --level to test your detector)
python -m botgame run-policy --model policy.pt --level 0.0
python -m botgame run-policy --model policy.pt --level 1.0 --interval 0.25
```

`--input-size` for `run-policy` must match the `--resize` used in
`build-dataset` (default 160x90). The model predicts an action type
(noop/tap/swipe) plus coordinates; humanization and telemetry are applied
exactly as in the dummy bot, so detector experiments stay comparable.

## Replay a recording and hunt bugs (Block 4)

The replicator re-injects the actions recovered from a gameplay video with the
original timing — record a session once, then repeat it loop after loop to
regression-test your game. While it plays, a health monitor checks for crashes
and ANRs (logcat crash buffer), the game losing foreground focus, and frozen
screens, saving a screenshot + JSONL record for every finding.

```bash
# From a getevent log (pixel-accurate), watching your game's package
python -m botgame replay --events events.log --src-size 1080 2400 \
    --package com.example.mygame --loops 10

# From a build-dataset output, or straight from a show-touches video
python -m botgame replay --dataset dataset --package com.example.mygame
python -m botgame replay --video play.mp4 --fps 10

# Stress variants: 2x speed, stop at the first anomaly, slight input noise
python -m botgame replay --events events.log --speed 2.0 --stop-on-anomaly
python -m botgame replay --events events.log --level 0.3 --seed 7 --loops 5
```

`--src-size` enables coordinate rescaling (target size is queried from the
device, or use `--dst-size`), so a recording from one phone replays on
another. Anomaly evidence lands in `bugreport/` (`report.jsonl` +
screenshots); the exit code is non-zero when anomalies were found, so the
command slots directly into CI against an emulator.

## Testing your detector (the red-team loop)

1. Run `run-dummy` across a sweep of `--level` (e.g. 0.0, 0.25, 0.5, 0.75, 1.0)
   with fixed `--seed` for reproducibility.
2. Have your detector label each session.
3. Plot detection rate vs. level → you get a curve showing which signals your
   detector relies on and where the gaps are.

## Architecture

```
capture (screencap)        -> botgame/adb/capture.py
  -> policy                   botgame/model/*, botgame/bots/policy.py
     or replay                botgame/replay.py     (recorded actions, exact timing)
  -> humanize                 botgame/humanize.py   (robotic <-> human knob)
  -> actuation (input)        botgame/adb/input.py
  -> telemetry                botgame/telemetry.py
  -> health monitor           botgame/monitor.py    (crash / focus / freeze)
```

## Roadmap

- [x] **Block 1** — ADB bridge + humanization + telemetry + dummy bot
- [x] **Block 2** — video → (state, action) dataset for imitation learning
- [x] **Block 3** — perception model + learned policy (behavioral cloning)
- [x] **Block 4** — deterministic replay + bug-hunting health monitor
- [ ] Faster capture (scrcpy/minicap) and minitouch backend for continuous gestures
- [ ] RL fine-tuning on top of the cloned policy
```
