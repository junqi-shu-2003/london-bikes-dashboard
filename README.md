# London Cycle Demand — Explore & Predict

English Dash application using the supplied London bikes data. Explore is implemented. Predict retrieves real weather, while hire predictions explicitly remain **Model pending** until coefficients are supplied. No model or weather API is required to run Explore.

## Run

```sh
uv sync
uv run python app.py
```

Open http://127.0.0.1:8050. Set `PORT` to change the port. `server = app.server` is available for a future gunicorn deployment. This project has not been deployed to Render.

## Data

`data/london_bikes.csv` is an unchanged copy of the user-supplied course file at `am01-code-sep2026-main/data/london_bikes.csv`.

The app reads its own local copy, never silently downloads or synthesizes values. Dates are calendar labels. The analysis period is 2014–2025, defaulting to 2025. There are 4,383 unique dates in this period and no missing values in hires or the five selected weather fields in the supplied snapshot. Weekday, weekend and season are derived from dates.

- All KPIs and charts use the inclusive selected date range.
- Average daily hires = sum of valid daily hire counts / valid observed days.
- Weekly (Monday–Sunday) and monthly series average selected observed days, including partial periods, not all days in the enclosing period.
- Weekday means show missing weekdays as missing rather than zero.
- Weather and colour selectors affect only the scatter; frequency affects only the trend.
- Hover shows date, values and group; Plotly provides zoom, legend selection and chart image download.
- Missing/infinite values are excluded per chart and counted in captions. Duplicate dates fail loading explicitly.
- Weather units follow the course helper convention; upstream measurement definitions still require checking before connecting a forecast model.

## Verify

```sh
uv run pytest -q
```

Tests cover known source totals, daily and partial-period aggregation, missing/invalid date ranges, one-day selections, every weather/group combination, and HTTP callback execution. Weather tests cover London dates, DST boundaries, cache persistence and expiry, manual refresh, failure fallback, incomplete input rejection, CSV export, bounded retries, and Predict callbacks. Dash implementation reference: https://dash.plotly.com/basic-callbacks

## Predict weather

- `open_meteo.py` requests Open-Meteo at a fixed central London point (51.5085, -0.1257), with `Europe/London` timezone and explicit dates and units.
- Future weather covers tomorrow through day five. The API request supplies exact `start_date` and `end_date`, so today is excluded.
- January 1–7, 2026 uses historical hourly reanalysis. Complete hourly coverage is validated before aggregation: precipitation is summed; the other variables are averaged.
- All output dates and five weather fields must be complete, finite and within basic physical bounds. Invalid results never replace a valid cache.
- Forecast TTL: one hour. History TTL: 24 hours. The page checks every minute while Predict is selected; fresh cache avoids a network call. Manual refresh bypasses TTL.
- Failed connections, timeouts, HTTP 429 and 5xx get at most one retry. A 60-second automatic failure cooldown avoids repeated requests. Manual refresh may retry sooner.
- Cache is in `.weather_cache/` (gitignored), with atomic file replacement and a memory fallback if writing is unavailable. Cache keys contain exact requested dates. Stale fallback retains its original fetch time and is clearly labelled. No matching cache means no weather values.
- Weather CSV downloads include fetch time, cache status, location, timezone and model status; no hire estimates are invented.
- Source: https://open-meteo.com/en/docs and https://open-meteo.com/en/docs/historical-weather-api . Fetch time is not a weather-model issue time. Historical reanalysis is not a genuine advance forecast backtest.

Live verification on 9 September 2026 returned five complete days for 10–14 September and seven complete days for 1–7 January. All 12 automated tests passed. This is a local app, not a deployed Render service; browser visual QA has not been performed.
