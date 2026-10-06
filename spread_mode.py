#!/usr/bin/env python3
"""Run one configured maker/taker cross-venue spread strategy."""

import argparse
import asyncio
import json
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import dotenv

from strategies.spread import SpreadConfig, SpreadStrategy


def load_config(path: Path, resume: bool) -> SpreadConfig:
    values = json.loads(path.read_text())
    allowed = {field.name for field in fields(SpreadConfig)} - {"resume"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown spread config fields: {sorted(unknown)}")
    for key in allowed & set(values):
        if key.endswith("_bps") or "fraction" in key or key in {
            "size", "max_position", "max_leverage", "max_slippage",
        }:
            values[key] = Decimal(str(values[key]))
    values["state_file"] = Path(values["state_file"])
    if values.get("observation_file") is not None:
        values["observation_file"] = Path(values["observation_file"])
    return SpreadConfig(**values, resume=resume)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--resume", action="store_true",
                        help="After manually reviewing both venue positions and the saved entry cost")
    args = parser.parse_args()
    if not args.env_file.is_file():
        raise FileNotFoundError(args.env_file)
    dotenv.load_dotenv(args.env_file)
    await SpreadStrategy(load_config(args.config, args.resume)).run()


if __name__ == "__main__":
    asyncio.run(main())
