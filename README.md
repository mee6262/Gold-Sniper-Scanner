# 🔱 Gold-Sniper-Scanner V2 — Hybrid Engine

V2 is the current version of the Gold Sniper project. The legacy V1 scanner has been removed from this branch; V1 is maintained separately.

## Architecture

- **`gold_sniper_v2.py`** — core multi-timeframe SMC candidate engine
- **`replay_test.py`** — historical replay / gate diagnostics / TP / expiry / management simulation
- **`mt5_data.py`** — MT5 market-data integration
- **`mt5_sniper_engine.py`** — MT5 sniper execution engine
- **`data_manager.py`** — data handling utilities
- **`runner_v2.py`** — V2 live runner
- **`.env.example`** — configuration template
- **`requirements.txt`** — Python dependencies

## V2 Logic

The engine analyzes **M30 + M15 + M5** and combines:

- M30 directional bias and zones
- M15 market-structure shift / sweep / FVG
- M5 MSS / FVG trigger
- location filtering
- score threshold
- RR validation
- deterministic Entry / SL / TP construction
- optional AI review after deterministic candidate generation

The replay engine is the main development/testing tool. It runs in safe mode without AI, Telegram, or live orders.

## Replay

Example:

```powershell
python replay_test.py
```

Replay output includes:

- gate diagnostics
- final candidates
- score buckets
- LONG / SHORT performance
- MFE / MAE
- trigger analysis
- TP simulation
- expiry simulation
- management simulation

## Current development target

The immediate objective is to make the V2 engine robust on the current **1,000 M5-bar replay window** before expanding validation to 10,000+ bars.

V1 baseline code is intentionally not kept in this repository because it is maintained as a separate project.
