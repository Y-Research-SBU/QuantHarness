# QuantAgent — Tournament Heartbeat

This file documents the periodic jobs that keep the tournament alive.

## Per-instance runners

The 4 paper-trading instances are launched independently:

* `run_continuous.py` — original baseline (already running, paper_trades.db)
* `run_instance.py --profile crypto_aggro` — paper_trades_crypto_aggro.db
* `run_instance.py --profile forex_focus`  — paper_trades_forex_focus.db
* `run_instance.py --profile top25_only`   — paper_trades_top25_only.db

Use `launch_tournament.sh` to bring up the 3 new ones; baseline stays
untouched. Each runner writes `instances/<name>/state.json` after every
scan cycle so the dashboard and judge can see live status.

## Data daemon (optional but recommended)

`python3 data_daemon.py` populates the Redis OHLCV cache. yfinance has
strict rate limits; running this once per host removes a large source
of pain. If Redis isn't reachable the daemon (and scanner) just fall
through to direct yfinance reads.

## Weekly tournament judge

`python3 tournament_judge.py` should run weekly. It computes 7-day
Sharpe per instance, picks the winner, and creates a new
`champion-YYYYMMDD` instance. Decisions are appended to
`tournament_log.json` and to `~/brain/quantagent-tournaments/`.

Suggested cron entry (Sunday 02:00 local):

```
0 2 * * 0 cd /Users/samib/Developer/quantagent && /usr/bin/env python3 tournament_judge.py
```

Or via OpenClaw's cron facility if available.

## Dashboard

`/` — main dashboard (with instance switcher)
`/tournament` — leaderboard + equity overlay + strategy attribution
`/api/instances` — JSON status of every instance
`/api/instance/<name>/portfolio` — DB-scoped per-instance view
`/api/tournament` — leaderboard payload (also pushed every 30s via socket.io)
