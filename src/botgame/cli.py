"""Command-line entrypoint: `python -m botgame <command>`.

Commands:
  devices                 List authorized USB devices.
  cap   [--out FILE]      Save a screenshot.
  tap    X Y              Inject a tap.
  swipe  X1 Y1 X2 Y2      Inject a swipe.
  run-dummy [...]         Run the placeholder bot against your game.
  build-dataset [...]     Turn a gameplay video (+events) into a dataset.
  train [...]             Train the imitation-learning policy on a dataset.
  run-policy [...]        Play live using a trained policy.
  replay [...]            Replicate recorded actions on-device and hunt bugs.
"""

from __future__ import annotations

import argparse
import sys
import time

from .adb import AdbDevice, ScreenCapture, TouchInput, list_devices
from .adb.device import AdbError
from .bots.dummy import DummyBot, DummyBotConfig
from .dataset import build_dataset, detect_touch, parse_getevent
from .dataset.schema import Action
from .humanize import Humanizer
from .telemetry import TelemetryLogger


def _device(args: argparse.Namespace) -> AdbDevice:
    if args.serial:
        return AdbDevice(serial=args.serial)
    return AdbDevice.autoconnect()


def cmd_devices(_: argparse.Namespace) -> int:
    devices = list_devices()
    if not devices:
        print("No authorized devices. Connect by USB and accept the RSA prompt.")
        return 1
    print("\n".join(devices))
    return 0


def cmd_cap(args: argparse.Namespace) -> int:
    cap = ScreenCapture(_device(args))
    cap.save(args.out)
    print(f"Saved screenshot to {args.out}")
    return 0


def cmd_tap(args: argparse.Namespace) -> int:
    TouchInput(_device(args)).tap(args.x, args.y)
    return 0


def cmd_swipe(args: argparse.Namespace) -> int:
    TouchInput(_device(args)).swipe(args.x1, args.y1, args.x2, args.y2, args.duration)
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
    with TelemetryLogger(out) as tel:
        bot = DummyBot(device, humanizer, tel, config)
        print(
            f"Running dummy bot: level={args.level} region={region or 'full'} "
            f"steps={args.steps or 'inf'} -> {out}  (Ctrl-C to stop)"
        )
        done = bot.run()
    print(f"Done. {done} actions logged to {out}")
    return 0


def _actions_from_overlay(video: str, fps: float) -> list[Action]:
    from .dataset.video import iter_frames

    actions: list[Action] = []
    for t, frame in iter_frames(video, target_fps=fps):
        hit = detect_touch(frame)
        if hit is not None:
            actions.append(Action.tap(t, hit[0], hit[1]))
    return actions


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from .dataset.video import iter_frames

    if args.events:
        with open(args.events, encoding="utf-8") as fh:
            src = tuple(args.src_size) if args.src_size else None
            dst = tuple(args.dst_size) if args.dst_size else None
            actions = parse_getevent(fh.read(), src_size=src, dst_size=dst)
        print(f"Parsed {len(actions)} actions from {args.events}")
    else:
        actions = _actions_from_overlay(args.video, args.fps)
        print(f"Recovered {len(actions)} taps from the show-touches overlay")

    resize = tuple(args.resize) if args.resize else None
    n = build_dataset(
        iter_frames(args.video, target_fps=args.fps),
        actions,
        args.out,
        fps=args.fps,
        resize=resize,
    )
    print(f"Dataset written to {args.out} ({n} samples)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .model.train import train

    train(
        args.dataset,
        tuple(args.screen_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
    )
    return 0


def cmd_run_policy(args: argparse.Namespace) -> int:
    from .bots.policy import PolicyBot, PolicyBotConfig

    device = _device(args)
    humanizer = Humanizer(level=args.level, seed=args.seed)
    config = PolicyBotConfig(
        interval_s=args.interval,
        max_steps=args.steps,
        input_size=tuple(args.input_size),
    )
    out = args.telemetry or f"telemetry/policy_{int(time.time())}.jsonl"
    with TelemetryLogger(out) as tel:
        bot = PolicyBot(device, args.model, humanizer, tel, config)
        print(
            f"Running policy {args.model}: level={args.level} "
            f"steps={args.steps or 'inf'} -> {out}  (Ctrl-C to stop)"
        )
        done = bot.run()
    print(f"Done. {done} steps logged to {out}")
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
        print("replay needs one of --events / --dataset / --video", file=sys.stderr)
        return 2
    if not actions:
        print(f"No actions recovered from {source}", file=sys.stderr)
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
        print(
            f"Replaying {len(actions)} actions from {source}: speed={args.speed}x "
            f"loops={args.loops} level={args.level} -> {out}  (Ctrl-C to stop)"
        )
        result = bot.run()

    print(
        f"Done. {result.actions_played} actions over {result.loops_completed} "
        f"loop(s){' (aborted)' if result.aborted else ''}; telemetry in {out}"
    )
    if result.anomalies:
        print(f"\n{len(result.anomalies)} anomalies found (details in {args.report}/):")
        for a in result.anomalies:
            first_line = a.detail.splitlines()[0] if a.detail else ""
            print(f"  - [{a.kind}] {first_line}" + (f"  ({a.screenshot})" if a.screenshot else ""))
        return 1
    if monitor is not None:
        print("No anomalies detected.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="botgame")
    parser.add_argument("-s", "--serial", help="ADB device serial (default: autodetect)")
    sub = parser.add_subparsers(dest="command", required=True)

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
    p_dummy.add_argument("--level", type=float, default=0.0, help="humanization 0..1")
    p_dummy.add_argument("--interval", type=float, default=0.8, help="seconds between actions")
    p_dummy.add_argument("--steps", type=int, default=50, help="0 = until Ctrl-C")
    p_dummy.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    p_dummy.add_argument("--swipes", action="store_true", help="use curved swipes")
    p_dummy.add_argument("--seed", type=int, default=None)
    p_dummy.add_argument("--telemetry", help="output JSONL path")
    p_dummy.set_defaults(func=cmd_run_dummy)

    p_ds = sub.add_parser("build-dataset", help="video (+events) -> dataset")
    p_ds.add_argument("--video", required=True, help="gameplay video path")
    p_ds.add_argument("--events", help="getevent -lt log (preferred over overlay)")
    p_ds.add_argument("--out", default="dataset", help="output directory")
    p_ds.add_argument("--fps", type=float, default=10.0, help="sampling rate")
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
    p_tr.add_argument("--out", default="policy.pt")
    p_tr.set_defaults(func=cmd_train)

    p_pol = sub.add_parser("run-policy", help="play live with a trained policy")
    p_pol.add_argument("--model", default="policy.pt", help="trained checkpoint")
    p_pol.add_argument("--level", type=float, default=0.0, help="humanization 0..1")
    p_pol.add_argument("--interval", type=float, default=0.2, help="seconds between steps")
    p_pol.add_argument("--steps", type=int, default=0, help="0 = until Ctrl-C")
    p_pol.add_argument("--input-size", type=int, nargs=2, metavar=("W", "H"), default=[160, 90],
                       help="must match dataset --resize used in training")
    p_pol.add_argument("--seed", type=int, default=None)
    p_pol.add_argument("--telemetry", help="output JSONL path")
    p_pol.set_defaults(func=cmd_run_policy)

    p_rep = sub.add_parser("replay", help="replicate recorded actions and hunt bugs")
    src = p_rep.add_argument_group("action source (pick one)")
    src.add_argument("--events", help="getevent -lt log")
    src.add_argument("--dataset", help="build-dataset output dir (uses labels.jsonl)")
    src.add_argument("--video", help="gameplay video (show-touches overlay fallback)")
    p_rep.add_argument("--fps", type=float, default=10.0, help="sampling rate for --video")
    p_rep.add_argument("--src-size", type=int, nargs=2, metavar=("W", "H"),
                       help="coord space of the recording (enables rescaling)")
    p_rep.add_argument("--dst-size", type=int, nargs=2, metavar=("W", "H"),
                       help="target screen px (default: queried from the device)")
    p_rep.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    p_rep.add_argument("--loops", type=int, default=1, help="repeat the sequence N times")
    p_rep.add_argument("--swipe-duration", type=int, default=250, help="ms per swipe")
    p_rep.add_argument("--level", type=float, default=0.0,
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AdbError as exc:
        print(f"ADB error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
