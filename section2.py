import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp, ttest_ind


ASSETS = dict(zip(
    "Gold Silver Platinum Palladium WTI Brent NaturalGas HeatingOil Copper Wheat Corn Soybeans Coffee Sugar Cotton".split(),
    "GC SI PL PA CL BZ NG HO HG ZW ZC ZS KC SB CT".split(),
    strict=True,
))
START, END = "2020-01-01", "2024-12-31"
WINDOW, STEP = 90, 30
ROWS, COLS = np.triu_indices(len(ASSETS), 1)

# Upper triangle transcribed from the two-decimal labels in Figure 1.
PAPER_ROWS = """
78 55 39 14 14 03 12 30 10 09 12 11 11 08
64 46 19 20 03 16 41 10 12 13 14 15 13
58 22 24 06 22 47 08 09 16 14 18 16
20 19 05 19 39 12 17 18 10 16 10
89 08 67 28 14 18 21 11 26 22
08 75 31 15 18 23 13 28 23
12 04 05 06 04 04 04 06
26 14 16 23 14 24 20
11 16 24 20 19 24
46 32 09 17 14
53 13 13 12
13 17 10
16 13
20
"""


def load_prices(raw):
    series, audit = {}, []
    for commodity, code in ASSETS.items():
        path = raw / f"{code}.json"
        payload = path.read_bytes()
        result, = json.loads(payload.decode("utf-8-sig"))["chart"]["result"]
        metadata = result["meta"]
        expected_type = {"FUTURE", "ALTSYMBOL"} if code == "CT" else {"FUTURE"}
        if metadata["symbol"] != f"{code}=F" or metadata["instrumentType"] not in expected_type:
            raise ValueError(f"Unexpected instrument in {path}")
        timezone = metadata["exchangeTimezoneName"]
        dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
        dates = dates.tz_convert(timezone).tz_localize(None).normalize()
        close = pd.Series(result["indicators"]["quote"][0]["close"], index=dates, dtype=float)
        close = close.sort_index().loc[START:END]
        if close.index.has_duplicates or close.notna().sum() < WINDOW + 1:
            raise ValueError(f"Invalid dates or incomplete series in {path}")
        series[commodity] = close
        audit.append({
            "commodity": commodity, "ticker": metadata["symbol"],
            "vendor_exchange": metadata["fullExchangeName"], "currency": metadata["currency"],
            "vendor_instrument_type": metadata["instrumentType"],
            "timezone": timezone, "rows": len(close), "null_rows": int(close.isna().sum()),
            "nonpositive_rows": int(close.le(0).sum()),
            "first_close": close.first_valid_index().date().isoformat(),
            "last_close": close.last_valid_index().date().isoformat(),
            "raw_file": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
        })
    panel = pd.DataFrame(series).sort_index()
    observed = panel.notna().any(axis=1)
    return panel.loc[observed], pd.DataFrame(audit), int((~observed).sum())


def calculate_returns(prices):
    log_prices = np.log(prices.where(prices.gt(0)))
    candidates = log_prices.diff()
    valid = candidates.notna().all(axis=1)
    previous = prices.shift()
    excluded = []
    for date in prices.index[~valid]:
        invalid = candidates.loc[date].isna()
        for commodity in invalid.index[invalid]:
            current, prior = prices.loc[date, commodity], previous.loc[date, commodity]
            excluded.append({
                "date": date, "commodity": commodity, "close": current, "previous_close": prior,
                "reason": "first_session" if date == prices.index[0] else
                "nonpositive_close" if current <= 0 or prior <= 0 else "missing_close",
            })
    return candidates.loc[valid], pd.DataFrame(excluded)


def correlation(returns):
    matrix = returns.corr().to_numpy(copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("A correlation is undefined. Check missing or constant returns.")
    np.fill_diagonal(matrix, 0)
    return matrix


def describe(matrix):
    edges = matrix[ROWS, COLS]
    absolute = np.abs(matrix)
    mean_signed, mean_absolute = edges.mean(), np.abs(edges).mean()
    eigen_signed = np.linalg.eigvalsh(matrix)[-1]
    eigen_absolute = np.linalg.eigvalsh(absolute)[-1]
    negative_mass = np.abs(edges[edges < 0]).sum()
    if not np.isclose(mean_absolute - mean_signed, 2 * negative_mass / len(edges), atol=1e-12):
        raise ValueError("Average-coupling identity failed")
    if eigen_absolute + 1e-12 < eigen_signed:
        raise ValueError("Largest-eigenvalue inequality failed")
    return {
        "mean_signed": mean_signed, "mean_absolute": mean_absolute,
        "mean_gap": mean_absolute - mean_signed,
        "lambda_signed": eigen_signed, "lambda_absolute": eigen_absolute,
        "lambda_gap": eigen_absolute - eigen_signed,
        "negative_edges": int((edges < 0).sum()),
        "negative_fraction": float((edges < 0).mean()),
        "negative_mass_fraction": negative_mass / np.abs(edges).sum(),
    }


def rolling(returns):
    rows, matrices = [], []
    for begin in range(0, len(returns) - WINDOW + 1, STEP):
        sample = returns.iloc[begin:begin + WINDOW]
        matrix = correlation(sample)
        matrices.append(matrix)
        rows.append({
            "window": len(rows) + 1, "start": sample.index[0], "end": sample.index[-1],
            "n": len(sample), **describe(matrix),
        })
    return pd.DataFrame(rows), np.stack(matrices)


def compare_fig1(matrix):
    labels = np.array(list(ASSETS))
    rows = [line.split() for line in PAPER_ROWS.strip().splitlines()]
    if [len(row) for row in rows] != list(range(14, 0, -1)):
        raise ValueError("Figure 1 reference rows have incorrect lengths")
    original = np.array([int(value) / 100 for row in rows for value in row])
    observed = matrix[ROWS, COLS]
    return pd.DataFrame({
        "asset_i": labels[ROWS], "asset_j": labels[COLS], "paper_figure_1": original,
        "replication": observed, "difference": observed - original,
        "absolute_difference": np.abs(observed - original),
        "matches_published_rounding": np.isclose(observed.round(2), original),
    })


def compare_paper_tests(windows):
    rows = []
    for metric, paper_ks, paper_ks_p, paper_welch_p, paper_g in [
        ("mean", 0.308, 0.049, 0.1906, 0.296),
        ("lambda", 0.128, 0.908, 0.5989, 0.118),
    ]:
        absolute = windows[f"{metric}_absolute"]
        signed = windows[f"{metric}_signed"]
        ks = ks_2samp(absolute, signed)
        welch = ttest_ind(absolute, signed, equal_var=False)
        df = len(absolute) + len(signed) - 2
        pooled_sd = np.sqrt((absolute.var(ddof=1) + signed.var(ddof=1)) / 2)
        hedges = (1 - 3 / (4 * df - 1)) * (absolute.mean() - signed.mean()) / pooled_sd
        for statistic, reference, value in [
            ("KS D", paper_ks, ks.statistic), ("KS nominal p", paper_ks_p, ks.pvalue),
            ("Welch nominal p", paper_welch_p, welch.pvalue), ("Hedges g", paper_g, hedges),
        ]:
            rows.append(dict(metric=metric, statistic=statistic, paper=reference,
                             replication=value, difference=value - reference))
    return pd.DataFrame(rows)


def make_plot(windows, output, figure_auditor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.13, right=0.975, top=0.91, bottom=0.11, hspace=0.35)
    for label, ax in zip("abc", axes, strict=True):
        ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(-25, 4),
                    textcoords="offset points", weight="bold", fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E6E8", linewidth=0.5)
    for ax, metric, title in zip(axes[:2], ("mean", "lambda"),
                                 ("Mean coupling", "Largest eigenvalue"), strict=True):
        ax.plot(windows.end, windows[f"{metric}_absolute"], color="#B75321", linewidth=1.25,
                linestyle="--", label="Absolute correlations")
        ax.plot(windows.end, windows[f"{metric}_signed"], color="#17678B", linewidth=1.25,
                label="Signed correlations")
        ax.set_ylabel(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.99),
               ncol=2, frameon=False)
    axes[2].plot(windows.end, 100 * windows.negative_fraction, color="#17678B", linewidth=1.25)
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("Negative edges (%)")
    axes[2].set_xlabel("Window end")
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if figure_auditor:
        import sys
        sys.path.insert(0, str(figure_auditor))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=output / "figure.alignment.json", strict=True,
        )
    fig.savefig(output / "rolling_baseline.pdf")
    fig.savefig(output / "rolling_baseline.svg")
    fig.savefig(output / "rolling_baseline.png", dpi=300)
    fig.savefig(output / "rolling_baseline.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replicate the published commodity Pearson baseline.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure-auditor", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prices, source, blank_dates = load_prices(args.raw)
    returns, excluded = calculate_returns(prices)
    windows, matrices = rolling(returns)
    matrix = correlation(returns)
    comparison = compare_fig1(matrix)
    datasets = {
        "close_panel": prices, "log_returns": returns, "source_audit": source,
        "excluded_returns": excluded, "rolling_baseline": windows,
        "figure1_comparison_105_pairs": comparison, "paper_statistics_comparison": compare_paper_tests(windows),
    }
    for name, data in datasets.items():
        indexed = name in {"close_panel", "log_returns"}
        data.to_csv(args.out / f"{name}.csv", index=indexed, index_label="date" if indexed else None)
    np.savez_compressed(args.out / "correlation_matrices.npz",
                        assets=np.array(list(ASSETS)), full_sample=matrix, signed=matrices,
                        absolute=np.abs(matrices),
                        window_end=windows.end.dt.strftime("%Y-%m-%d").to_numpy(dtype=str))
    summary = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "window": WINDOW, "step": STEP,
        "quote_dates": len(prices), "all_blank_dates_removed": blank_dates,
        "valid_returns": len(returns), "windows": len(windows),
        "first_return": str(returns.index[0].date()), "last_return": str(returns.index[-1].date()),
        "first_window_end": str(windows.end.iloc[0].date()),
        "last_window_end": str(windows.end.iloc[-1].date()),
        "full_sample": describe(matrix),
        "figure1_matches_2dp": int(comparison.matches_published_rounding.sum()),
        "figure1_mae": float(comparison.absolute_difference.mean()),
        "figure1_max_difference": float(comparison.absolute_difference.max()),
        "figure1_across_pairs_r": float(comparison.paper_figure_1.corr(comparison.replication)),
        "rolling": {},
    }
    for metric in ["mean_signed", "mean_absolute", "mean_gap", "lambda_signed", "lambda_absolute",
                   "lambda_gap", "negative_edges", "negative_fraction", "negative_mass_fraction"]:
        values = windows[metric]
        summary["rolling"][metric] = {
            "min": float(values.min()), "mean": float(values.mean()),
            "median": float(values.median()), "max": float(values.max()),
            "max_date": str(windows.end.iloc[values.idxmax()].date()),
        }
    summary["windows_with_negative_edges"] = int(windows.negative_edges.gt(0).sum())
    summary["rolling_mean_r"] = float(windows.mean_signed.corr(windows.mean_absolute))
    summary["rolling_lambda_r"] = float(windows.lambda_signed.corr(windows.lambda_absolute))
    result = json.dumps(summary, indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(result + "\n", encoding="utf-8")
    make_plot(windows, args.out, args.figure_auditor)
    print(result)


if __name__ == "__main__":
    main()
