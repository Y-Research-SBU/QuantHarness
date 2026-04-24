"""
Live-trading continuous runner.

Mirror of :mod:`run_continuous` but swaps :class:`PaperTradingEngine`
for :class:`HyperliquidExecutor` so signals are routed to a real DEX.

Safety model
------------
* **Testnet by default.** Mainnet requires the explicit ``--live`` flag *and*
  an interactive ``I UNDERSTAND`` confirmation. There is deliberately no env
  var to suppress the confirmation — this is not a gate worth paving over.
* ``HYPERLIQUID_PRIVATE_KEY`` must be set in the environment; the key never
  touches argv or logs.
* Safety caps (max position size, max daily loss) are set on the executor
  itself and enforced per-trade on mainnet.

CLI::

    python run_live.py --symbols BTC-USD ETH-USD --once        # testnet, one cycle
    python run_live.py --live --symbols BTC-USD                # mainnet (prompts)
    python run_live.py --live --yes ...  # skip the prompt (CI / automated)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from hyperliquid_config import (
    MAINNET_CONFIRMATION_PHRASE,
    MAX_DAILY_LOSS_USD_MAINNET,
    MAX_POSITION_USD_MAINNET,
    PRIVATE_KEY_ENV_VAR,
)
from hyperliquid_executor import HyperliquidExecutor, HyperliquidExecutorError
from market_config import MARKETS
from run_continuous import ContinuousRunner
from scanner import MarketScanner

logger = logging.getLogger(__name__)


# ── ANSI colour helpers for the warning banner ──────────────────────
_RED = "\033[31;1m"
_YELLOW = "\033[33;1m"
_CYAN = "\033[36;1m"
_RESET = "\033[0m"


def _print_banner(live: bool, symbols: List[str], max_pos: float, max_loss: float) -> None:
    """Emit a loud, scannable banner so the operator can't miss what mode we're in."""
    network = "MAINNET — REAL MONEY" if live else "TESTNET (paper)"
    colour = _RED if live else _CYAN
    bar = "═" * 72
    print(colour + bar + _RESET)
    print(colour + f" QuantAgent LIVE RUNNER — {network} ".center(72) + _RESET)
    print(colour + bar + _RESET)
    print(f" Symbols:           {', '.join(symbols)}")
    print(f" Max position size: ${max_pos:,.2f}")
    print(f" Max daily loss:    ${max_loss:,.2f}")
    if live:
        print(_YELLOW + " ⚠  Trades are real. SL/TP are enforced server-side." + _RESET)
    print(colour + bar + _RESET)


def _confirm_mainnet(skip: bool) -> bool:
    """Interactive confirmation for mainnet. Returns True to proceed."""
    if skip:
        logger.warning("Mainnet confirmation bypassed via --yes")
        return True
    phrase = MAINNET_CONFIRMATION_PHRASE
    print(_RED + f"\nType '{phrase}' to enable mainnet trading (anything else aborts):" + _RESET)
    try:
        response = input("> ").strip()
    except EOFError:
        response = ""
    if response != phrase:
        print("Aborted.")
        return False
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    parser = argparse.ArgumentParser(
        description="QuantAgent LIVE trading runner (Hyperliquid). Testnet by default.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="⚠  Use MAINNET. Requires interactive confirmation.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the mainnet confirmation prompt (CI/automation only).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=default_symbols,
        help=f"Symbols to trade. Default: {' '.join(default_symbols)}",
    )
    parser.add_argument("--db", default=None, help="Path to SQLite database (optional).")
    parser.add_argument(
        "--summary-every-minutes",
        type=int,
        default=30,
        help="Minutes between performance summaries.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after N scan cycles (useful for smoke tests).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan per market and exit.",
    )
    parser.add_argument(
        "--no-kronos",
        action="store_true",
        help="Disable Kronos forecasting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build everything, run safety checks, but skip actual order submission.",
    )
    parser.add_argument(
        "--max-position-usd",
        type=float,
        default=None,
        help=f"Cap per-position notional. Mainnet default: ${MAX_POSITION_USD_MAINNET:.2f}",
    )
    parser.add_argument(
        "--max-daily-loss-usd",
        type=float,
        default=None,
        help=f"Kill switch on daily realized P&L. Mainnet default: ${MAX_DAILY_LOSS_USD_MAINNET:.2f}",
    )
    return parser


def _filter_supported(symbols: List[str]) -> List[str]:
    """Drop symbols Hyperliquid doesn't list (from hyperliquid_config.SYMBOL_MAP)."""
    from hyperliquid_config import SYMBOL_MAP
    keep: List[str] = []
    for s in symbols:
        if s not in MARKETS:
            logger.warning("Unknown market %s — skipping", s)
            continue
        if s not in SYMBOL_MAP:
            logger.warning("%s has no Hyperliquid mapping — skipping", s)
            continue
        keep.append(s)
    return keep


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    symbols = _filter_supported(args.symbols)
    if not symbols:
        print("No supported symbols to trade. Exiting.", file=sys.stderr)
        return 2

    testnet = not args.live

    if not os.environ.get(PRIVATE_KEY_ENV_VAR):
        print(
            f"Error: {PRIVATE_KEY_ENV_VAR} is not set — required for signing.",
            file=sys.stderr,
        )
        return 2

    _print_banner(
        live=args.live,
        symbols=symbols,
        max_pos=args.max_position_usd if args.max_position_usd is not None else MAX_POSITION_USD_MAINNET,
        max_loss=args.max_daily_loss_usd if args.max_daily_loss_usd is not None else MAX_DAILY_LOSS_USD_MAINNET,
    )

    if args.live and not _confirm_mainnet(skip=args.yes):
        return 1

    try:
        executor = HyperliquidExecutor(
            testnet=testnet,
            db_path=args.db,
            max_position_usd=args.max_position_usd,
            max_daily_loss_usd=args.max_daily_loss_usd,
            dry_run=args.dry_run,
        )
    except HyperliquidExecutorError as exc:
        print(f"Executor setup failed: {exc}", file=sys.stderr)
        return 2

    summary = executor.get_account_summary()
    logger.info(
        "Account ready — value=$%.2f margin_used=$%.2f withdrawable=$%.2f",
        summary["account_value"], summary["total_margin_used"], summary["withdrawable"],
    )

    scanner = MarketScanner(db_path=args.db, use_kronos=not args.no_kronos)
    # The scanner constructs its own PaperTradingEngine by default; replace it.
    scanner.engine = executor
    runner = ContinuousRunner(
        scanner=scanner,
        symbols=symbols,
        summary_every_seconds=args.summary_every_minutes * 60,
    )
    runner.install_signal_handlers()

    if args.once:
        runner.run_once()
    else:
        runner.run(max_cycles=args.max_cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
