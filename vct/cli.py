"""Command line interface for interacting with the RoboDog brain."""

from __future__ import annotations

import argparse
import json
import sys

from .configuration import overrides_from_iter
from .robodog.dog_bot_brain import RoboDogBrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with the RoboDog brain controller")
    parser.add_argument("--config", default="vct/config.yaml", help="Path to configuration file")
    parser.add_argument("--wav", help="Run recognition on an audio file")
    parser.add_argument("--cmd", help="Command text to process")
    parser.add_argument("--gpio-pin", type=int, default=None, help="GPIO pin for reward actuator")
    parser.add_argument("--simulate", action="store_true", help="Run without accessing hardware")
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override configuration values without editing the file. "
            "Supports dotted keys, e.g. --set weights.stimulus=0.55"
        ),
    )
    args = parser.parse_args()

    try:
        overrides = overrides_from_iter(args.overrides)
    except ValueError as exc:  # pragma: no cover - defensive branch
        parser.error(str(exc))
        sys.exit(2)

    brain = RoboDogBrain(
        cfg_path=args.config,
        gpio_pin=args.gpio_pin,
        simulate=args.simulate,
        config_overrides=overrides,
    )
    if args.wav:
        result = brain.run_once_from_wav(args.wav)
    else:
        result = brain.handle_command(args.cmd or "сидіти")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
