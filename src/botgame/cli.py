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
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

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
    with contextlib.ExitStack() as stack:
        tel = stack.enter_context(TelemetryLogger(out))
        capture = _open_capture(stack, device, args.fast)
        touch = _open_touch(stack, device, args.minitouch, args.minitouch_port)
        bot = DummyBot(device, humanizer, tel, config, capture=capture, touch=touch)
        print(
            f"Running dummy bot: level={args.level} region={region or 'full'} "
            f"steps={args.steps or 'inf'} capture={'fast' if args.fast else 'screencap'} "
            f"touch={'minitouch' if args.minitouch else 'input'} "
            f"-> {out}  (Ctrl-C to stop)"
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
    print("level,score")
    for r in results:
        print(f"{r.level:.2f},{r.score:.4f}")
    print(f"\nWrote {len(results)} rows to {args.out}/results.csv")
    if args.plot:
        from .redteam.plot import plot_sweep
        png = plot_sweep(
            os.path.join(args.out, "results.csv"),
            os.path.join(args.out, "plot.png"),
        )
        print(f"Wrote plot to {png}")
    return 0


def cmd_redteam_plot(args: argparse.Namespace) -> int:
    from .redteam.plot import plot_sweep

    out = args.out or os.path.splitext(args.csv)[0] + ".png"
    plot_sweep(args.csv, out, title=args.title)
    print(f"Wrote plot to {out}")
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
    print(f"Saved RL checkpoint to {args.out}")
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
    with contextlib.ExitStack() as stack:
        tel = stack.enter_context(TelemetryLogger(out))
        capture = _open_capture(stack, device, args.fast)
        touch = _open_touch(stack, device, args.minitouch, args.minitouch_port)
        bot = PolicyBot(
            device, args.model, humanizer, tel, config,
            capture=capture, touch=touch,
        )
        print(
            f"Running policy {args.model}: level={args.level} "
            f"steps={args.steps or 'inf'} capture={'fast' if args.fast else 'screencap'} "
            f"touch={'minitouch' if args.minitouch else 'input'} "
            f"-> {out}  (Ctrl-C to stop)"
        )
        done = bot.run()
    print(f"Done. {done} steps logged to {out}")
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

    p_rt = sub.add_parser("redteam", help="sweep humanization, score with your detector")
    p_rt.add_argument("--levels", type=float, nargs="+",
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
    p_pol.add_argument("--level", type=float, default=0.0, help="humanization 0..1")
    p_pol.add_argument("--interval", type=float, default=0.2, help="seconds between steps")
    p_pol.add_argument("--steps", type=int, default=0, help="0 = until Ctrl-C")
    p_pol.add_argument("--input-size", type=int, nargs=2, metavar=("W", "H"), default=[160, 90],
                       help="must match dataset --resize used in training")
    p_pol.add_argument("--seed", type=int, default=None)
    p_pol.add_argument("--telemetry", help="output JSONL path")
    p_pol.add_argument("--fast", action="store_true",
                       help="streaming H.264 capture (requires PyAV)")
    p_pol.add_argument("--minitouch", action="store_true",
                       help="route touches via minitouch TCP socket")
    p_pol.add_argument("--minitouch-port", type=int, default=1111,
                       help="local TCP port forwarded to localabstract:minitouch")
    p_pol.set_defaults(func=cmd_run_policy)

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
