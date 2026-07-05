"""Command-line entrypoint: `python -m botgame <command>`.

Commands:
  devices                 List authorized USB devices.
  cap   [--out FILE]      Save a screenshot.
  tap    X Y              Inject a tap.
  swipe  X1 Y1 X2 Y2      Inject a swipe.
  run-dummy [...]         Run the placeholder bot against your game.
  build-dataset [...]     Turn a gameplay video (+events) into a dataset.
  train [...]             Train the imitation-learning policy on a dataset.
  learn [...]             One shot: video (+events) -> dataset -> trained policy.
  run-policy [...]        Play live using a trained policy.
  replay [...]            Replicate recorded actions on-device and hunt bugs.
  redteam [...]           Sweep humanization, score with your detector.
  redteam-plot [...]      Render a redteam sweep CSV as a detection-curve PNG.
  train-rl [...]          PPO fine-tune the behavioral-cloning policy.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import textwrap
import time

from . import __version__
from .adb import (
    AdbDevice,
    FastCapture,
    MiniTouch,
    ScreenCapture,
    TouchInput,
    list_devices,
)
from .adb.device import AdbError
from .bots.dummy import DummyBot, DummyBotConfig
from .dataset import build_dataset, detect_touch, parse_getevent
from .dataset.schema import Action
from .humanize import Humanizer
from .telemetry import TelemetryLogger

# --- terminal presentation (TTY-aware, zero deps) ---------------------------
# Colour only when stdout is a real terminal and the user hasn't opted out via
# the conventional NO_COLOR env var, so piped/redirected output stays clean.

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_CODES = {
    "bold": "\033[1m", "dim": "\033[2m", "green": "\033[32m",
    "yellow": "\033[33m", "red": "\033[31m", "cyan": "\033[36m", "reset": "\033[0m",
}


def _c(text: str, *styles: str) -> str:
    if not _COLOR:
        return text
    prefix = "".join(_CODES[s] for s in styles if s in _CODES)
    return f"{prefix}{text}{_CODES['reset']}" if prefix else text


def ok(msg: str) -> None:
    print(f"{_c('✓', 'green', 'bold')} {msg}")


def info(msg: str) -> None:
    print(f"{_c('→', 'cyan')} {msg}")


def warn(msg: str) -> None:
    print(f"{_c('⚠', 'yellow', 'bold')} {msg}", file=sys.stderr)


def err(msg: str) -> None:
    print(f"{_c('✗', 'red', 'bold')} {msg}", file=sys.stderr)


def hint(msg: str) -> None:
    print(f"  {_c(msg, 'dim')}")


# --- argument validators (friendly errors instead of downstream crashes) ----

def _ratio(value: str) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0 (got {v})")
    return v


def _positive(value: str) -> float:
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0 (got {v})")
    return v


# --- backends ---------------------------------------------------------------

def _open_capture(stack: contextlib.ExitStack, device: AdbDevice, fast: bool):
    if fast:
        return stack.enter_context(FastCapture(device))
    return ScreenCapture(device)


def _open_touch(stack: contextlib.ExitStack, device: AdbDevice, minitouch: bool, port: int):
    if minitouch:
        return stack.enter_context(MiniTouch(device, host_port=port))
    return TouchInput(device)


def _device(args: argparse.Namespace) -> AdbDevice:
    if args.serial:
        return AdbDevice(serial=args.serial)
    return AdbDevice.autoconnect()


def cmd_devices(_: argparse.Namespace) -> int:
    devices = list_devices()
    if not devices:
        warn("No authorized devices found.")
        hint("Connect the phone by USB, enable USB debugging, and accept the RSA prompt.")
        hint("Then check `adb devices` lists it as 'device'.")
        return 1
    ok(f"{len(devices)} device(s) ready:")
    for serial in devices:
        print(f"  • {serial}")
    return 0


def cmd_cap(args: argparse.Namespace) -> int:
    cap = ScreenCapture(_device(args))
    cap.save(args.out)
    ok(f"Screenshot saved to {_c(args.out, 'bold')}")
    return 0


def cmd_tap(args: argparse.Namespace) -> int:
    TouchInput(_device(args)).tap(args.x, args.y)
    ok(f"Tapped ({args.x}, {args.y})")
    return 0


def cmd_swipe(args: argparse.Namespace) -> int:
    TouchInput(_device(args)).swipe(args.x1, args.y1, args.x2, args.y2, args.duration)
    ok(f"Swiped ({args.x1}, {args.y1}) → ({args.x2}, {args.y2}) in {args.duration}ms")
    return 0


def cmd_run_dummy(args: argparse.Namespace) -> int:
    device = _device(args)
    humanizer = Humanizer(level=args.level, seed=args.seed)
    region = tuple(args.region) if args.region else None
    config = DummyBotConfig(
        region=region,
        interval_s=args.interval,
        max_steps=args.steps,
        use_swipes=args.swipes,
        seed=args.seed,
    )
    out = args.telemetry or f"telemetry/dummy_{int(time.time())}.jsonl"
    with contextlib.ExitStack() as stack:
        tel = stack.enter_context(TelemetryLogger(out))
        capture = _open_capture(stack, device, args.fast)
        touch = _open_touch(stack, device, args.minitouch, args.minitouch_port)
        bot = DummyBot(device, humanizer, tel, config, capture=capture, touch=touch)
        info(
            f"Dummy bot · level={args.level} · region={region or 'full'} · "
            f"steps={args.steps or '∞'} · capture={'fast' if args.fast else 'screencap'} · "
            f"touch={'minitouch' if args.minitouch else 'input'}"
        )
        hint(f"logging to {out}   (Ctrl-C to stop)")
        done = bot.run()
    ok(f"Done — {done} actions logged to {out}")
    return 0


def _actions_from_overlay(video: str, fps: float) -> list[Action]:
    from .dataset.video import iter_frames

    actions: list[Action] = []
    for t, frame in iter_frames(video, target_fps=fps):
        hit = detect_touch(frame)
        if hit is not None:
            actions.append(Action.tap(t, hit[0], hit[1]))
    return actions


def _recover_actions(events: str | None, video: str, fps: float,
                     src_size, dst_size) -> list[Action]:
    """Recover the action stream from a getevent log or a show-touches video."""
    if events:
        with open(events, encoding="utf-8") as fh:
            src = tuple(src_size) if src_size else None
            dst = tuple(dst_size) if dst_size else None
            actions = parse_getevent(fh.read(), src_size=src, dst_size=dst)
        info(f"Parsed {len(actions)} actions from {events}")
    else:
        actions = _actions_from_overlay(video, fps)
        info(f"Recovered {len(actions)} taps from the show-touches overlay")
    return actions


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from .dataset.video import iter_frames

    actions = _recover_actions(args.events, args.video, args.fps,
                               args.src_size, args.dst_size)
    resize = tuple(args.resize) if args.resize else None
    n = build_dataset(
        iter_frames(args.video, target_fps=args.fps),
        actions,
        args.out,
        fps=args.fps,
        resize=resize,
    )
    ok(f"Dataset written to {_c(args.out, 'bold')} ({n} samples)")
    hint(f"next:  python -m botgame train --dataset {args.out} --screen-size W H")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .model.train import train

    train(
        args.dataset,
        tuple(args.screen_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stack=args.stack,
        val_split=args.val_split,
        out_path=args.out,
    )
    ok(f"Policy trained → {_c(args.out, 'bold')}")
    hint(f"next:  python -m botgame run-policy --model {args.out}")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """One shot: gameplay video (+events) -> dataset -> trained policy."""
    from .dataset.video import iter_frames
    from .model.train import train

    actions = _recover_actions(args.events, args.video, args.fps,
                               args.src_size, args.dst_size)
    if not actions:
        err("No actions recovered; nothing to learn from.")
        hint("Record touch events with `adb shell getevent -lt > events.log`, or "
             "enable Developer options → 'Show taps' before recording.")
        return 1

    if args.screen_size:
        screen_size = tuple(args.screen_size)
    elif args.dst_size:
        screen_size = tuple(args.dst_size)
    else:
        # The action coords live in screen px == video px; read the first frame.
        _, first = next(iter(iter_frames(args.video, target_fps=args.fps)))
        screen_size = (first.shape[1], first.shape[0])
        info(f"Screen size inferred from video: {screen_size[0]}x{screen_size[1]}")

    n = build_dataset(
        iter_frames(args.video, target_fps=args.fps),
        actions,
        args.dataset,
        fps=args.fps,
        resize=tuple(args.resize),
    )
    ok(f"Dataset written to {args.dataset} ({n} samples)")

    print()
    train(
        args.dataset,
        screen_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stack=args.stack,
        val_split=args.val_split,
        out_path=args.out,
    )
    ok(f"Learned policy → {_c(args.out, 'bold')}")
    hint(f"now let it play:  python -m botgame run-policy --model {args.out}")
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from .redteam import load_detector, run_sweep, sample_detector

    device = _device(args) if not args.dry_run else None
    levels = args.levels

    if args.detector:
        detector = load_detector(args.detector)
    else:
        detector = sample_detector

    def factory(level: float, seed: int, telemetry_path: str) -> str:
        humanizer = Humanizer(level=level, seed=seed)
        config = DummyBotConfig(
            interval_s=args.interval,
            max_steps=args.steps,
            seed=seed,
            capture_frames=not args.dry_run,
        )
        with contextlib.ExitStack() as stack:
            tel = stack.enter_context(TelemetryLogger(telemetry_path))
            if args.dry_run:
                # Simulated session: emit deterministic telemetry without a device.
                for step in range(args.steps):
                    tx, ty = 500, 500
                    delay = humanizer.reaction_delay()
                    jx, jy = humanizer.jitter_point(tx, ty)
                    tel.log("tap", target=[tx, ty], actual=[jx, jy],
                            reaction_s=round(delay, 4), level=level, step=step)
            else:
                bot = DummyBot(device, humanizer, tel, config)
                bot.run()
        return telemetry_path

    results = run_sweep(
        factory, detector, levels,
        seed_base=args.seed_base,
        sessions_per_level=args.sessions_per_level,
        out_dir=args.out,
    )
    # Plain, machine-parseable table (kept uncoloured on purpose).
    print("level,score")
    for r in results:
        print(f"{r.level:.2f},{r.score:.4f}")
    ok(f"Wrote {len(results)} rows to {args.out}/results.csv")
    if args.plot:
        from .redteam.plot import plot_sweep
        png = plot_sweep(
            os.path.join(args.out, "results.csv"),
            os.path.join(args.out, "plot.png"),
        )
        ok(f"Plot written to {png}")
    return 0


def cmd_redteam_plot(args: argparse.Namespace) -> int:
    from .redteam.plot import plot_sweep

    out = args.out or os.path.splitext(args.csv)[0] + ".png"
    plot_sweep(args.csv, out, title=args.title)
    ok(f"Plot written to {_c(out, 'bold')}")
    return 0


def cmd_train_rl(args: argparse.Namespace) -> int:
    from .rl import PPOConfig, RandomEnv, train_rl

    if args.env == "random":
        env = RandomEnv(seed=args.seed)
    else:
        raise SystemExit(
            "Live BotEnv is game-specific; instantiate it in Python with your "
            "reward function. Only --env random is wired through the CLI."
        )

    config = PPOConfig(
        rollout_steps=args.rollout_steps,
        epochs=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        screen_size=tuple(args.screen_size),
        input_size=tuple(args.input_size),
    )

    def _log(stats):
        print(
            f"iter={stats.epoch_index} r={stats.mean_reward:+.3f} "
            f"v={stats.mean_value:+.3f} pi={stats.policy_loss:+.4f} "
            f"vf={stats.value_loss:.4f} H={stats.entropy:.3f} "
            f"t={stats.elapsed_s:.2f}s"
        )

    history = train_rl(
        env, out_path=args.out, total_steps=args.steps,
        bc_checkpoint=args.bc, config=config,
    )
    for s in history:
        _log(s)
    ok(f"RL checkpoint saved → {_c(args.out, 'bold')}")
    return 0


def _print_anomalies(anomalies: list, report_dir: str) -> None:
    warn(f"{len(anomalies)} anomaly(ies) found — evidence in {report_dir}/")
    for a in anomalies:
        first_line = a.detail.splitlines()[0] if a.detail else ""
        loc = f"  ({a.screenshot})" if a.screenshot else ""
        print(f"  {_c('•', 'yellow')} [{a.kind}] {first_line}{loc}")


def cmd_run_policy(args: argparse.Namespace) -> int:
    from .bots.policy import PolicyBot, PolicyBotConfig
    from .monitor import HealthMonitor

    device = _device(args)
    humanizer = Humanizer(level=args.level, seed=args.seed)
    config = PolicyBotConfig(
        interval_s=args.interval,
        max_steps=args.steps,
        input_size=tuple(args.input_size),
        act_threshold=args.threshold,
        check_every=args.check_every,
        stop_on_anomaly=args.stop_on_anomaly,
    )
    monitor = None
    if not args.no_monitor:
        monitor = HealthMonitor(device, package=args.package, report_dir=args.report)

    out = args.telemetry or f"telemetry/policy_{int(time.time())}.jsonl"
    with contextlib.ExitStack() as stack:
        tel = stack.enter_context(TelemetryLogger(out))
        capture = _open_capture(stack, device, args.fast)
        touch = _open_touch(stack, device, args.minitouch, args.minitouch_port)
        bot = PolicyBot(
            device, args.model, humanizer, tel, config,
            capture=capture, touch=touch, monitor=monitor,
        )
        info(
            f"Policy {os.path.basename(args.model)} · level={args.level} · "
            f"threshold={args.threshold} · steps={args.steps or '∞'} · "
            f"capture={'fast' if args.fast else 'screencap'} · "
            f"touch={'minitouch' if args.minitouch else 'input'}"
        )
        hint(f"logging to {out}   (Ctrl-C to stop)")
        done = bot.run()
    ok(f"Done — {done} steps logged to {out}")
    if bot.anomalies:
        _print_anomalies(bot.anomalies, args.report)
        return 1
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .monitor import HealthMonitor
    from .replay import ReplayBot, ReplayConfig, load_actions_jsonl, rescale_actions

    if args.events:
        with open(args.events, encoding="utf-8") as fh:
            actions = parse_getevent(fh.read())
        source = args.events
    elif args.dataset:
        source = f"{args.dataset}/labels.jsonl"
        actions = load_actions_jsonl(source)
    elif args.video:
        actions = _actions_from_overlay(args.video, args.fps)
        source = args.video
    else:
        err("replay needs one action source: --events, --dataset or --video")
        return 2
    if not actions:
        err(f"No actions recovered from {source}")
        return 1

    device = _device(args)
    if args.src_size:
        dst = tuple(args.dst_size) if args.dst_size else device.screen_size()
        actions = rescale_actions(actions, tuple(args.src_size), dst)

    monitor = None
    if not args.no_monitor:
        monitor = HealthMonitor(
            device,
            package=args.package,
            report_dir=args.report,
            freeze_checks=args.freeze_checks,
        )

    humanizer = Humanizer(level=args.level, seed=args.seed)
    config = ReplayConfig(
        speed=args.speed,
        loops=args.loops,
        swipe_duration_ms=args.swipe_duration,
        check_every=args.check_every,
        stop_on_anomaly=args.stop_on_anomaly,
    )
    out = args.telemetry or f"telemetry/replay_{int(time.time())}.jsonl"
    with TelemetryLogger(out) as tel:
        bot = ReplayBot(TouchInput(device), actions, humanizer, tel, monitor, config)
        info(
            f"Replaying {len(actions)} actions from {source} · "
            f"speed={args.speed}x · loops={args.loops} · level={args.level}"
        )
        hint(f"logging to {out}   (Ctrl-C to stop)")
        result = bot.run()

    ok(
        f"Done — {result.actions_played} actions over {result.loops_completed} "
        f"loop(s){' (aborted)' if result.aborted else ''}; telemetry in {out}"
    )
    if result.anomalies:
        _print_anomalies(result.anomalies, args.report)
        return 1
    if monitor is not None:
        ok("No anomalies detected.")
    return 0


_EPILOG = textwrap.dedent(
    """\
    quick start
      1. python -m botgame devices                       # confirm the phone is connected
      2. python -m botgame learn --video play.mp4 \\
             --events events.log --out policy.pt         # learn to play from a recording
      3. python -m botgame run-policy --model policy.pt \\
             --package com.example.mygame                # let it play + hunt bugs

    Run `python -m botgame <command> --help` for the options of any command.
    Set NO_COLOR=1 to disable coloured output.
    """
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="botgame",
        description="PC-driven Android game bot: learn from a recording, play "
                    "autonomously, and red-team your own bot detector.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-s", "--serial", help="ADB device serial (default: autodetect)")
    parser.add_argument("--version", action="version", version=f"botgame {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("devices", help="list authorized devices").set_defaults(func=cmd_devices)

    p_cap = sub.add_parser("cap", help="save a screenshot")
    p_cap.add_argument("--out", default="captures/screen.png")
    p_cap.set_defaults(func=cmd_cap)

    p_tap = sub.add_parser("tap", help="inject a tap")
    p_tap.add_argument("x", type=int)
    p_tap.add_argument("y", type=int)
    p_tap.set_defaults(func=cmd_tap)

    p_swipe = sub.add_parser("swipe", help="inject a swipe")
    p_swipe.add_argument("x1", type=int)
    p_swipe.add_argument("y1", type=int)
    p_swipe.add_argument("x2", type=int)
    p_swipe.add_argument("y2", type=int)
    p_swipe.add_argument("--duration", type=int, default=200, help="ms")
    p_swipe.set_defaults(func=cmd_swipe)

    p_dummy = sub.add_parser("run-dummy", help="run the placeholder bot")
    p_dummy.add_argument("--level", type=_ratio, default=0.0, help="humanization 0..1")
    p_dummy.add_argument("--interval", type=float, default=0.8, help="seconds between actions")
    p_dummy.add_argument("--steps", type=int, default=50, help="0 = until Ctrl-C")
    p_dummy.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    p_dummy.add_argument("--swipes", action="store_true", help="use curved swipes")
    p_dummy.add_argument("--seed", type=int, default=None)
    p_dummy.add_argument("--telemetry", help="output JSONL path")
    p_dummy.add_argument("--fast", action="store_true",
                         help="streaming H.264 capture (requires PyAV)")
    p_dummy.add_argument("--minitouch", action="store_true",
                         help="route touches via minitouch TCP socket")
    p_dummy.add_argument("--minitouch-port", type=int, default=1111,
                         help="local TCP port forwarded to localabstract:minitouch")
    p_dummy.set_defaults(func=cmd_run_dummy)

    p_ds = sub.add_parser("build-dataset", help="video (+events) -> dataset")
    p_ds.add_argument("--video", required=True, help="gameplay video path")
    p_ds.add_argument("--events", help="getevent -lt log (preferred over overlay)")
    p_ds.add_argument("--out", default="dataset", help="output directory")
    p_ds.add_argument("--fps", type=_positive, default=10.0, help="sampling rate")
    p_ds.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=[160, 90])
    p_ds.add_argument("--src-size", type=int, nargs=2, metavar=("W", "H"),
                      help="touch-device coord space (for getevent rescaling)")
    p_ds.add_argument("--dst-size", type=int, nargs=2, metavar=("W", "H"),
                      help="screen size in px (for getevent rescaling)")
    p_ds.set_defaults(func=cmd_build_dataset)

    p_tr = sub.add_parser("train", help="train the imitation-learning policy")
    p_tr.add_argument("--dataset", default="dataset", help="build-dataset output dir")
    p_tr.add_argument("--screen-size", type=int, nargs=2, metavar=("W", "H"), required=True,
                      help="screen px the dataset coords are in")
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--stack", type=int, default=4,
                      help="frames of temporal context per sample (1 = single frame)")
    p_tr.add_argument("--val-split", type=_ratio, default=0.1,
                      help="tail fraction held out for validation metrics")
    p_tr.add_argument("--out", default="policy.pt")
    p_tr.set_defaults(func=cmd_train)

    p_lr = sub.add_parser("learn", help="one shot: video (+events) -> trained policy")
    p_lr.add_argument("--video", required=True, help="gameplay video path")
    p_lr.add_argument("--events", help="getevent -lt log (preferred over overlay)")
    p_lr.add_argument("--fps", type=_positive, default=10.0, help="sampling rate")
    p_lr.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=[160, 90])
    p_lr.add_argument("--src-size", type=int, nargs=2, metavar=("W", "H"),
                      help="touch-device coord space (for getevent rescaling)")
    p_lr.add_argument("--dst-size", type=int, nargs=2, metavar=("W", "H"),
                      help="screen size in px (for getevent rescaling)")
    p_lr.add_argument("--screen-size", type=int, nargs=2, metavar=("W", "H"),
                      help="coord space override (default: video frame size)")
    p_lr.add_argument("--dataset", default="dataset", help="where to write the dataset")
    p_lr.add_argument("--epochs", type=int, default=20)
    p_lr.add_argument("--batch-size", type=int, default=32)
    p_lr.add_argument("--lr", type=float, default=1e-3)
    p_lr.add_argument("--stack", type=int, default=4,
                      help="frames of temporal context per sample")
    p_lr.add_argument("--val-split", type=_ratio, default=0.1)
    p_lr.add_argument("--out", default="policy.pt")
    p_lr.set_defaults(func=cmd_learn)

    p_rt = sub.add_parser("redteam", help="sweep humanization, score with your detector")
    p_rt.add_argument("--levels", type=_ratio, nargs="+",
                      default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p_rt.add_argument("--steps", type=int, default=50,
                      help="bot steps per level")
    p_rt.add_argument("--interval", type=float, default=0.4)
    p_rt.add_argument("--seed-base", type=int, default=0)
    p_rt.add_argument("--detector",
                      help="'module:function' returning a bot probability in [0,1]; "
                           "omit to use the built-in sample detector")
    p_rt.add_argument("--sessions-per-level", type=int, default=1,
                      help="how many seeded sessions to run at each level")
    p_rt.add_argument("--out", default="redteam", help="output directory")
    p_rt.add_argument("--dry-run", action="store_true",
                      help="synthesise telemetry without a device (CI / smoke test)")
    p_rt.add_argument("--plot", action="store_true",
                      help="also write a PNG plot (requires matplotlib)")
    p_rt.set_defaults(func=cmd_redteam)

    p_rtp = sub.add_parser("redteam-plot",
                           help="render the sweep CSV as a detection-curve PNG")
    p_rtp.add_argument("csv", help="path to a sweep results.csv")
    p_rtp.add_argument("--out", help="output PNG (default: alongside the CSV)")
    p_rtp.add_argument("--title", help="optional plot title")
    p_rtp.set_defaults(func=cmd_redteam_plot)

    p_rl = sub.add_parser("train-rl", help="PPO fine-tune the BC policy")
    p_rl.add_argument("--env", default="random", choices=["random"],
                      help="only 'random' is wired via CLI; build BotEnv in Python")
    p_rl.add_argument("--bc", help="behavioral-cloning checkpoint to warm-start from")
    p_rl.add_argument("--out", default="policy_rl.pt")
    p_rl.add_argument("--steps", type=int, default=512)
    p_rl.add_argument("--rollout-steps", type=int, default=64)
    p_rl.add_argument("--epochs-per-update", type=int, default=4)
    p_rl.add_argument("--minibatch-size", type=int, default=32)
    p_rl.add_argument("--lr", type=float, default=3e-4)
    p_rl.add_argument("--screen-size", type=int, nargs=2, metavar=("W", "H"),
                      default=[1080, 2400])
    p_rl.add_argument("--input-size", type=int, nargs=2, metavar=("W", "H"),
                      default=[160, 90])
    p_rl.add_argument("--seed", type=int, default=None)
    p_rl.set_defaults(func=cmd_train_rl)

    p_pol = sub.add_parser("run-policy", help="play live with a trained policy")
    p_pol.add_argument("--model", default="policy.pt", help="trained checkpoint")
    p_pol.add_argument("--level", type=_ratio, default=0.0, help="humanization 0..1")
    p_pol.add_argument("--interval", type=float, default=0.2, help="seconds between steps")
    p_pol.add_argument("--steps", type=int, default=0, help="0 = until Ctrl-C")
    p_pol.add_argument("--threshold", type=_ratio, default=0.5,
                       help="min confidence to act; below it the bot waits (0 = always act)")
    p_pol.add_argument("--input-size", type=int, nargs=2, metavar=("W", "H"), default=[160, 90],
                       help="fallback for old checkpoints without stored input size")
    p_pol.add_argument("--seed", type=int, default=None)
    p_pol.add_argument("--package", help="app package to watch for lost focus / crashes")
    p_pol.add_argument("--report", default="bugreport", help="anomaly evidence dir")
    p_pol.add_argument("--check-every", type=int, default=10,
                       help="health-check every N steps (0 = never)")
    p_pol.add_argument("--stop-on-anomaly", action="store_true",
                       help="stop playing on the first anomaly")
    p_pol.add_argument("--no-monitor", action="store_true", help="disable health checks")
    p_pol.add_argument("--telemetry", help="output JSONL path")
    p_pol.add_argument("--fast", action="store_true",
                       help="streaming H.264 capture (requires PyAV)")
    p_pol.add_argument("--minitouch", action="store_true",
                       help="route touches via minitouch TCP socket")
    p_pol.add_argument("--minitouch-port", type=int, default=1111,
                       help="local TCP port forwarded to localabstract:minitouch")
    p_pol.set_defaults(func=cmd_run_policy)

    p_rep = sub.add_parser("replay", help="replicate recorded actions and hunt bugs")
    src = p_rep.add_argument_group("action source (pick one)")
    src.add_argument("--events", help="getevent -lt log")
    src.add_argument("--dataset", help="build-dataset output dir (uses labels.jsonl)")
    src.add_argument("--video", help="gameplay video (show-touches overlay fallback)")
    p_rep.add_argument("--fps", type=_positive, default=10.0, help="sampling rate for --video")
    p_rep.add_argument("--src-size", type=int, nargs=2, metavar=("W", "H"),
                       help="coord space of the recording (enables rescaling)")
    p_rep.add_argument("--dst-size", type=int, nargs=2, metavar=("W", "H"),
                       help="target screen px (default: queried from the device)")
    p_rep.add_argument("--speed", type=_positive, default=1.0, help="playback speed multiplier")
    p_rep.add_argument("--loops", type=int, default=1, help="repeat the sequence N times")
    p_rep.add_argument("--swipe-duration", type=int, default=250, help="ms per swipe")
    p_rep.add_argument("--level", type=_ratio, default=0.0,
                       help="humanization 0..1 (0 = exact replica)")
    p_rep.add_argument("--seed", type=int, default=None)
    p_rep.add_argument("--package", help="app package to watch for lost focus / crashes")
    p_rep.add_argument("--report", default="bugreport", help="anomaly evidence dir")
    p_rep.add_argument("--check-every", type=int, default=5,
                       help="health-check every N actions (0 = only at the end)")
    p_rep.add_argument("--freeze-checks", type=int, default=3,
                       help="consecutive identical frames before reporting a freeze")
    p_rep.add_argument("--stop-on-anomaly", action="store_true",
                       help="abort the replay on the first anomaly")
    p_rep.add_argument("--no-monitor", action="store_true", help="disable health checks")
    p_rep.add_argument("--telemetry", help="output JSONL path")
    p_rep.set_defaults(func=cmd_replay)

    return parser


# Map a failed lazy import to the pip package that provides it.
_PIP_HINT = {
    "torch": "torch",
    "cv2": "opencv-python",
    "av": "av",
    "matplotlib": "matplotlib",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        info("Interrupted.")
        return 130
    except AdbError as exc:
        err(f"ADB error: {exc}")
        return 2
    except FileNotFoundError as exc:
        target = exc.filename or exc
        err(f"File not found: {target}")
        return 2
    except ImportError as exc:
        name = getattr(exc, "name", None) or ""
        pkg = _PIP_HINT.get(name, name or "the missing dependency")
        err(f"Missing optional dependency '{name}'." if name else f"Missing dependency: {exc}")
        hint(f"install it with:  pip install {pkg}")
        return 3
    except ValueError as exc:
        err(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
