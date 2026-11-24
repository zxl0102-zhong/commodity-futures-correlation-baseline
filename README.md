# Commodity Futures Correlation Baseline

This repository reproduces the return-correlation baseline in Georgescu and Kinnunen (2026), [Entropy 28, 344](https://doi.org/10.3390/e28030344). It uses Yahoo Finance closes for 15 commodity futures series during 2020–2024, computes complete-case daily log returns, and compares signed Pearson matrices with their elementwise absolute values over 90 trading-day windows advanced in steps of 30 trading days. The 30-day step is taken from the paper's MI/PDI specification.

## Run

    python -m pip install -r requirements.txt
    python download.py --out yahoo
    python section2.py --raw yahoo --out results

The baseline was verified with Python 3.14.7. The analysis can also use an archived directory containing the 15 Yahoo Finance chart JSON files named by code, for example GC.json and CL.json. The optional download flag --translate-relay routes the public chart request through Google Translate. Direct access can return HTTP 429. Data and results may differ if Yahoo revises historical quotes.

The script writes the aligned closes and returns, requested and excluded date audits, 39 rolling matrix pairs and their summary statistics, comparisons with published Figure 1 and Table 3, and the rolling baseline plot. With the 2026-11-24 snapshot, there were 1,254 common return dates and 39 rolling windows. Source hashes and the date-level exclusion log are written with each run.

The original figure's reported correlations are transcribed to two decimals for numerical comparison. Published KS and Welch statistics are reproduced as numerical checks. Their nominal p-values do not account for pairing and overlapping windows.
