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
- **Block 4** — **fast capture**: streaming H.264 from `screenrecord` decoded
  in a background thread (~30 fps vs 1-5 fps for per-frame `screencap`). Wire
  it into the bots with `--fast`.
- **Block 5** — **minitouch**: high-frequency TCP-streamed multitouch input
  (continuous curved gestures, per-finger pressure). `--minitouch` on the
  bots; `MiniTouch` class drops in wherever `TouchInput` was used.
- **Block 6** — **automated red-team loop**: `botgame redteam` sweeps
  `--level` 0 → 1, runs the bot at each, calls a pluggable detector, and
  writes a CSV of (level, detection score). Plot the curve to see where your
  detector breaks.
- **Block 7** — **PPO fine-tuning**: actor-critic head over the BC trunk; load
  the BC checkpoint to warm-start. Ships with a `RandomEnv` stub for smoke
  tests; plug your game-specific reward into `BotEnv` for live training.

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

## Fast capture (Block 4)

Default capture uses `adb exec-out screencap -p` — one PNG per frame, capped
around 1-5 fps. For real-time play, pass `--fast` to switch to a streaming
backend: `adb exec-out screenrecord --output-format=h264 -` writes H.264 to
stdout, decoded by PyAV in a background thread. `grab()` returns the latest
frame (~30 fps).

```bash
pip install av
python -m botgame run-dummy  --fast --level 0.5 --interval 0.1
python -m botgame run-policy --fast --model policy.pt --level 0.5 --interval 0.05
```

`screenrecord` caps each session at 180 s on most Android builds; `FastCapture`
auto-restarts the subprocess before that, so the stream runs indefinitely.

## Fast input via minitouch (Block 5)

Default touch uses `adb shell input` (one JVM-backed process per gesture).
Pass `--minitouch` to stream taps and curved gestures over a TCP socket to a
minitouch daemon running on the device. The daemon must be pushed and started
once before each session:

```bash
adb push minitouch /data/local/tmp/minitouch
adb shell chmod 755 /data/local/tmp/minitouch
adb shell /data/local/tmp/minitouch &      # leave running

python -m botgame run-dummy  --minitouch --level 0.5 --interval 0.1
python -m botgame run-policy --minitouch --model policy.pt --level 0.5
```

`MiniTouch` parses the daemon's banner (max_x / max_y / max_pressure / pid)
and rescales screen-pixel coordinates to touch-panel coordinates on the fly,
so the rest of the pipeline keeps speaking screen pixels.

## Automated red-team sweep (Block 6)

`botgame redteam` runs one session per `--level`, calls your detector on the
resulting telemetry, and writes a CSV summary. Bring your own detector via
`--detector module:function` (signature: `(telemetry_path) -> float in [0,1]`)
or omit it to use the built-in `sample_detector` heuristic.

```bash
# smoke test without a device (synthesises telemetry from the humanizer):
python -m botgame redteam --dry-run --levels 0.0 0.25 0.5 0.75 1.0 \
    --steps 50 --out redteam/

# real run against your game and your detector:
python -m botgame redteam --levels 0.0 0.25 0.5 0.75 1.0 --steps 200 \
    --detector my_pkg.detector:score
```

Plot `level` vs `score` from `redteam/results.csv` to see at which humanization
level your detector starts missing the bot.

## RL fine-tuning (Block 7)

Once you have a BC checkpoint, fine-tune it with PPO against a game-specific
reward. The `train-rl` CLI only exposes the `RandomEnv` smoke test; for real
training, build a `BotEnv` in Python with your capture + touch + reward:

```python
from botgame.adb import AdbDevice, FastCapture, MiniTouch
from botgame.rl import BotEnv, PPOConfig, train_rl

device = AdbDevice.autoconnect()
with FastCapture(device) as cap, MiniTouch(device) as touch:
    env = BotEnv(cap, touch, reward_fn=my_reward, step_interval_s=0.1)
    train_rl(env, "policy_rl.pt", total_steps=5_000,
             bc_checkpoint="policy.pt", config=PPOConfig())
```

`my_reward(prev_frame, action, next_frame)` is where you implement the game
score signal — OCR on the score HUD, pixel-template matching on the win
banner, whatever fits.

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
- [x] **Block 2** — video → (state, action) dataset for imitation learning
- [x] **Block 3** — perception model + learned policy (behavioral cloning)
- [x] **Block 4** — streaming H.264 capture (`--fast`) via screenrecord + PyAV
- [x] **Block 5** — minitouch backend for continuous, multitouch gestures (`--minitouch`)
- [x] **Block 6** — automated red-team loop (`botgame redteam`)
- [x] **Block 7** — PPO fine-tuning on top of the cloned policy (`botgame train-rl`)
```
