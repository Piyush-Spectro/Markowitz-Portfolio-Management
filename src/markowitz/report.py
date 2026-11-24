"""Export auditable tables and a portable research report."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from .portfolio import covariance, efficient_frontier

COLORS = ["#778899", "#267d91", "#ce8539", "#6055a5"]


def write_report(result, training_returns, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["returns", "weights", "trades", "forecasts", "metrics"]:
        value = getattr(result, name)
        value.to_csv(output / f"{name}.csv", index=name in {"returns", "metrics"})
    (output / "metadata.json").write_text(json.dumps({**metadata, "config": result.config}, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#334155",
                         "text.color": "#172b3a", "axes.titleweight": "bold", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    wealth = (1 + result.returns).cumprod()
    for (name, series), color in zip(wealth.items(), COLORS):
        # Include pre-investment wealth in drawdown, including a first-day loss.
        peak = np.maximum.accumulate(np.r_[1.0, series.to_numpy()])[1:]
        axes[0, 0].plot(series.index, series, label=name, color=color, lw=1.7)
        axes[0, 1].plot(series.index, series / peak - 1, color=color, lw=1.4)
    axes[0, 0].set(title="Growth of 1 unit of capital", ylabel="Net wealth")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(title="Drawdown from prior peak", ylabel="Drawdown")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    snapshot = result.weights.groupby("strategy", sort=False).tail(1).set_index("strategy").drop(columns="execution_date")
    snapshot.T.plot.bar(ax=axes[1, 0], color=COLORS, width=0.8, legend=False)
    axes[1, 0].set(title="Last rebalance allocations", ylabel="Portfolio weight", xlabel="")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].tick_params(axis="x", rotation=35)
    mu = np.asarray(training_returns).mean(axis=0) * result.config["annualization"]
    cov = covariance(training_returns, result.config["annualization"])
    frontier = efficient_frontier(mu, cov, max_weight=result.config["max_weight"])
    frontier.to_csv(output / "frontier.csv", index=False)
    axes[1, 1].plot(frontier.volatility, frontier["return"], color=COLORS[1], lw=2.5)
    axes[1, 1].set(title="Initial training-window efficient frontier", xlabel="Annualized volatility",
                   ylabel="Annualized expected return")
    axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle("MARKOWITZ / Portfolio research\n" + metadata["data_label"], fontsize=17, ha="left", x=0.02)
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)
    metrics = result.metrics.copy()
    for col in metrics:
        metrics[col] = metrics[col].map(lambda value: f"{value:.2f}" if col == "sharpe" else f"{value:.1%}")
    metrics.columns = ["Total return", "CAGR", "Volatility", "Sharpe", "Max drawdown"]
    error = result.forecasts.predicted_daily_return - result.forecasts.realized_daily_return
    forecast_rmse = float(np.sqrt(np.mean(error ** 2)))
    fallback_count = int(result.trades.min_variance_fallback.sum())
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Markowitz | Research report</title><style>
body{margin:0;background:#f4f6f8;color:#172b3a;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:56px 24px}
h1{font-size:42px;line-height:1.12;letter-spacing:-1.4px}h2{margin-top:36px;font-size:22px}.eyebrow{letter-spacing:2px;font-size:12px;color:#267d91;font-weight:700}
.card{background:white;border:1px solid #dce3e9;border-radius:12px;padding:24px;margin:24px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;text-align:right;border-bottom:1px solid #e2e8f0}th:first-child{text-align:left}.scroll{overflow-x:auto}code{background:#e8eef2;padding:2px 5px}p{max-width:900px}.muted{color:#586777}@media(max-width:600px){main{padding:24px 12px}h1{font-size:32px}.card{padding:14px}}
</style><main><div class="eyebrow">RESEARCH NOTE / REPRODUCIBLE EXPERIMENT</div><h1>Forecasts are inputs.<br>Portfolio performance is the test.</h1>'''
    page += f'<p>{html.escape(metadata["data_label"])}. Model: <strong>{html.escape(result.config["forecast_method"])}</strong>.</p>'
    page += '<div class="card"><h2>Walk-forward results, after trading costs</h2><div class="scroll">' + metrics.to_html(escape=True, border=0) + '</div></div>'
    page += '<div class="card"><img src="overview.png" alt="Net wealth, drawdown, allocation and training-only efficient frontier"></div>'
    page += f'<h2>What this run measures</h2><p>Test period: {result.returns.index.min().date()} to {result.returns.index.max().date()}. '
    page += f'Train window: {result.config["train_window"]} observations. Rebalance interval: {result.config["rebalance"]} observations. '
    page += f'Trading cost: {result.config["cost_bps"]:g} basis points per unit bought or sold. Initial investment is charged. '
    page += f'Risk-free assumption: {result.config["risk_free"]:.1%} annually. Forecast RMSE: {forecast_rmse:.6f} in daily-return units.</p>'
    page += f'<p>There were {fallback_count} minimum-variance fallbacks where a positive expected excess return was infeasible. '
    page += 'All rebalances, costs, forecasts and target weights are included in the adjacent CSV files.</p>'
    page += '<h2>Interpretation and limits</h2><p>The forecast estimates a holding-period average return. A good forecast score does not guarantee a better allocation. '
    page += 'Concentration, covariance estimates and trading costs can outweigh a small predictive advantage. '
    page += 'The frontier is a training-sample illustration, not a forecast of realized performance.</p>'
    page += '<p>The included sample is synthetic. Its results establish that the pipeline runs; they do not establish investment skill. '
    page += 'For real-market experiments, use point-in-time adjusted prices and document universe selection, missing observations and corporate actions. '
    page += 'This implementation assumes daily observations, close-price execution, full investment, no shorting, proportional costs and no taxes or market impact.</p>'
    page += '<p class="muted">Source and configuration fingerprints are recorded in metadata.json. Educational research; no investment recommendation.</p></main></html>'
    (output / "report.html").write_text(page)
