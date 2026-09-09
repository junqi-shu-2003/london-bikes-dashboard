# London Cycle Demand

English Dash dashboard with Explore, Predict and Models pages. Group 2's final Model E coefficients are connected. This local revision includes model comparison; upload it to GitHub to update the existing Render service.

## Run and verify

```sh
uv sync
uv run python app.py
uv run pytest -q
```

Local URL: http://127.0.0.1:8050/ . The 17 automated tests cover source totals, aggregation, callbacks, all six models against reproduced fitted values, the final coefficient calculation, missing inputs, cache/failure handling and 23/25-hour London days.

## Explore

The unchanged course dataset is `data/london_bikes.csv`. Explore covers 4,383 dates in 2014–2025, defaults to 2025 and retains zero-hire days. KPIs use observed daily counts; weekly/monthly charts average selected observed days, including partial periods. Missing values are not filled with zero. Scatter associations are not causal effects.

## Models

Six notebook specifications (Baseline, A–E) are reproduced on the same 4,381 rows: dates from 2014 onward and hires greater than 1. This excludes two zero-hire days. Models compares adjusted R², residual standard error, 2025 RMSE/MAE, observed/modelled monthly averages and daily residuals. Filters affect diagnostics, not the fixed summary metrics.

The 2025 check fits each formula on 2014–2024 (4,016 rows) and evaluates 2025 (365 rows) using observed weather. Because formulas were selected with knowledge of the full dataset, this is a retrospective temporal check, not an untouched test or a test of advance weather forecasts. Model E changes the specification: D and E are not nested.

`model_coefficients.csv` is the supplied final export, unchanged and used for E inference. Its coefficients match the notebook formula reproduced from the course data within 4.8e-11. Runtime verifies its SHA256 against the audit bundle. Other model coefficients and audit metadata are in `data/model_comparison.json`; historical chart values are in `data/model_fitted.csv` and `data/model_holdout.csv`. No refitting occurs during dashboard interaction.

To deliberately rebuild audited outputs after a model change:

```sh
uv run python scripts/build_models.py
```

The script contains reviewed formulas; it does not execute the supplied notebook. Statsmodels is only a development dependency. The original notebook is retained locally in `model_sources/` and excluded from the deployment package.

## Predict and weather

- Predict selects a model and overlays alternatives for the same weather. Final E uses temperature, precipitation, wind, visibility (km), solar energy (MJ/m²), weekday, season, temperature × precipitation and a post-12-Sep-2022 indicator. Monday/Autumn are baselines.
- The training wind definition remains unconfirmed. Choose daily maximum or mean explicitly. D/E need this choice; earlier models do not. Neither convention is silently assumed.
- Open-Meteo location is central London (51.5085, -0.1257). Tomorrow through day five excludes today, using the Europe/London date.
- Hourly API requests use GMT Unix timestamps with coverage padding. Aggregation maps timestamps to London dates and validates every required hour, including DST days.
- Temperature, humidity, dew point, visibility and cloud use means; precipitation is summed; temperature/wind maxima are retained separately. Hourly shortwave radiation integrates to daily MJ/m².
- January 1–7, 2026 uses historical reanalysis, supplemented with visibility from the historical forecast archive. This mixed source is labelled. It is not an advance-forecast backtest, and observed January hires are unavailable here.
- Missing required inputs stop affected model predictions. No input imputation or negative-prediction clipping. Training-range exceedances and negative estimates are flagged. Estimates are point predictions without uncertainty intervals.
- Forecast cache TTL is one hour; historical TTL is 24 hours. Refresh bypasses TTL. Network errors, 429 and 5xx receive one retry; automated failures have a 60-second cooldown. Matching stale cache retains its original timestamp and is labelled; failed requests never replace valid cache.
- `.weather_cache/` uses atomic writes with memory fallback. Render's ephemeral filesystem may discard disk cache on restarts. CSV exports include predictions, weather, source, fetch time and wind convention.

References: [Open-Meteo forecast](https://open-meteo.com/en/docs), [historical weather](https://open-meteo.com/en/docs/historical-weather-api), [historical forecast](https://open-meteo.com/en/docs/historical-forecast-api), [Visual Crossing field definitions](https://www2.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/). Original weather provider provenance still needs confirmation.

## Render

Python Web Service, repository root, Python 3.13.5:

Build: `pip install uv && uv sync --frozen --no-dev`

Start: `uv run --no-sync gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`

Keep all runtime Python modules, coefficient CSV, data files and assets together. This revision has been tested locally; uploading a package alone does not mean a Render deployment succeeded.
