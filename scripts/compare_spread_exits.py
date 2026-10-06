#!/usr/bin/env python3
"""Compare indicative center and opposite-band exits on recorded basis samples.

This is a signal comparison. It does not assume a maker order filled or claim
that the sampled bid/ask was executable for the desired size.
"""

import argparse
import json
import statistics
from decimal import Decimal
from pathlib import Path

from spread_mode import load_config
from strategies.spread_model import BPS, closing_signal, opening_signal, opposite_band_exit


def _net_bps(config, entry_row, exit_row, direction, days):
    entry_quotes = entry_row.get("quotes")
    exit_quotes = exit_row.get("quotes")
    if entry_quotes is not None or exit_quotes is not None:
        if entry_quotes is None or exit_quotes is None:
            return None
        maker_direction = direction if config.maker_exchange == config.venue_a else -direction
        entry_side = "buy" if maker_direction > 0 else "sell"
        exit_side = "sell" if maker_direction > 0 else "buy"
        entry = entry_quotes.get(entry_side)
        exit_quote = exit_quotes.get(exit_side)
        if entry is None or exit_quote is None:
            return None
        if Decimal(exit_quote["quantity"]) < Decimal(entry["quantity"]):
            return None
        maker_mid = (Decimal(exit_row["maker_bid"]) + Decimal(exit_row["maker_ask"])) / 2
        hedge_mid = (Decimal(exit_row["hedge_bid"]) + Decimal(exit_row["hedge_ask"])) / 2
        reference = max(maker_mid, hedge_mid)
        cash = (Decimal(entry["cash_per_unit"]) - Decimal(entry["fees_per_unit"])
                + Decimal(exit_quote["cash_per_unit"]) - Decimal(exit_quote["fees_per_unit"])
                - reference * config.carry_reserve_bps * days / BPS)
        return cash * BPS / reference
    # Compatibility with older observations containing only the indicative mid.
    gross = direction * (Decimal(exit_row["basis_bps"]) - Decimal(entry_row["basis_bps"]))
    costs = (2 * (config.maker_fee_bps + config.taker_fee_bps)
             + config.exit_reserve_bps + config.carry_reserve_bps * days)
    return gross - costs


def _entry_eligible(config, row, direction):
    if "margin_buffer" in row and Decimal(row["margin_buffer"]) <= config.warning_buffer_fraction:
        return False
    quotes = row.get("quotes")
    if quotes is None:
        return True
    maker_direction = direction if config.maker_exchange == config.venue_a else -direction
    side = "buy" if maker_direction > 0 else "sell"
    quote = quotes.get(side)
    if quote is None:
        return False
    executed_basis = Decimal(quote["basis_bps"])
    edge = (config.center_bps - executed_basis if direction > 0 else
            executed_basis - config.center_bps)
    required = (config.maker_fee_bps + config.taker_fee_bps + config.exit_reserve_bps
                + config.carry_reserve_bps + config.min_profit_bps)
    return edge > required


def compare(config, rows):
    previous = 0
    entries = []
    for index, row in enumerate(rows):
        direction = opening_signal(Decimal(row["basis_bps"]), config.center_bps, config.entry_band_bps)
        if direction and direction != previous and _entry_eligible(config, row, direction):
            entries.append((index, direction))
        previous = direction

    result = {}
    for rule in ("center", "opposite_band"):
        net_bps = []
        durations = []
        for start, direction in entries:
            for row in rows[start + 1:]:
                basis = Decimal(row["basis_bps"])
                ready = (closing_signal(basis, config.center_bps, config.exit_band_bps, direction)
                         if rule == "center" else
                         opposite_band_exit(basis, config.center_bps, config.entry_band_bps, direction))
                if ready:
                    days = max(Decimal(1), Decimal(str(row["time"] - rows[start]["time"])) / Decimal(86400))
                    net = _net_bps(config, rows[start], row, direction, days)
                    if net is None:
                        continue
                    net_bps.append(net)
                    durations.append(float(days))
                    break
        result[rule] = {
            "entry_signals": len(entries), "completed_signals": len(net_bps),
            "positive_after_assumed_costs": sum(value > 0 for value in net_bps),
            "median_indicative_net_bps": str(statistics.median(net_bps)) if net_bps else None,
            "median_holding_days": statistics.median(durations) if durations else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, resume=False)
    rows = [json.loads(line) for line in args.observations.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["time"])
    print(json.dumps(compare(config, rows), indent=2))


if __name__ == "__main__":
    main()
