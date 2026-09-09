# Group 2 model integration audit

Reproduced on 9 September 2026 from the supplied course data and reviewed notebook formulas. Source notebook and final coefficient export remain unchanged.

| Model | Adjusted R² | Residual SE | 2025 RMSE | 2025 MAE |
|---|---:|---:|---:|---:|
| Baseline | 0.362382 | 7321.73 | 7330.55 | 5856.29 |
| A | 0.413469 | 7022.29 | 6775.62 | 5208.27 |
| B | 0.455195 | 6767.90 | 7384.29 | 5910.55 |
| C | 0.499645 | 6485.93 | 7167.90 | 5771.20 |
| D | 0.575313 | 5975.41 | 7644.88 | 6059.42 |
| E | 0.678234 | 5201.20 | 5005.42 | 3803.05 |

Full-sample fits use 4,381 days. The retrospective temporal comparison uses 4,016 training days (2014–2024) and 365 evaluation days (2025), with observed weather. Specifications were selected using the full dataset: this is not an untouched test. E has the lowest retrospective RMSE among these six specifications, not a guarantee of future accuracy.

Final E coefficient reproduction maximum absolute difference: 4.729372449219227e-11. The runtime checks the original coefficient file hash and reproduces fitted values for all six models.

## Interpretation corrections

- Notebook numeric VIF values without an intercept include temperature 6.232, wind 5.397 and visibility 5.966. The accompanying claim that all values are below 5 is inconsistent with its output. The dashboard shows this original calculation separately from a full design-matrix VIF including the intercept, categorical variables, interaction and price indicator (intercept omitted from displayed results).
- E removes humidity, maximum temperature and dew point from D, and adds other terms. D/E are not nested models.
- Temperature's marginal association is 656.70 − 21.41 × precipitation, because E has an interaction. The +656.70 coefficient alone applies when precipitation is zero.
- The post-price-change indicator is an association with a date period, not proof of a causal price effect.
- Two zero-hire days are excluded to reproduce the notebook. The notebook's explanation for these days is not independently verified; Explore retains the original observations.

## Weather comparability

The original wind definition/provider remains to be confirmed. Both daily maximum and daily mean are available, but D/E require an explicit selection. January visibility is from archived forecasts while other January weather is reanalysis. This is disclosed in the interface. Future weather and all seven January days returned complete extended fields in local live checks on 9 September 2026. Remote Render operation of this revision remains to be verified after deployment.
