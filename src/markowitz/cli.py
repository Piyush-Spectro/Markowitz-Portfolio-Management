"""Command-line entry point for a complete, auditable experiment."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import sys
from .backtest import Config, run_backtest
from .data import load_prices, returns_from_prices
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("data/sample_prices.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--model", choices=["historical", "ridge", "cnn_bilstm", "lstm", "bilstm"], default="ridge")
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--risk-free", type=float, default=0.02)
    parser.add_argument("--cost-bps", type=float, default=10)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        prices = load_prices(args.prices)
        config = Config(train_window=args.train_window, rebalance=args.rebalance, risk_free=args.risk_free,
                        cost_bps=args.cost_bps, max_weight=args.max_weight, top_k=args.top_k,
                        forecast_method=args.model, neural_epochs=args.epochs, seed=args.seed)
        result = run_backtest(prices, config)
        versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "matplotlib"]}
        if args.model in {"cnn_bilstm", "lstm", "bilstm"}:
            versions["torch"] = importlib.metadata.version("torch")
        metadata = {"data_label": "Synthetic demonstration / not market performance" if all(str(c).startswith("Asset_") for c in prices.columns)
                    else "User-supplied prices / provenance must be verified by the researcher",
                    "input_file": args.prices.name, "sha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
                    "observations": len(prices), "assets": list(prices.columns), "python": sys.version.split()[0], "versions": versions}
        write_report(result, returns_from_prices(prices).iloc[:config.train_window], args.output, metadata)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(result.metrics.round(4).to_string())
    print(f"\nReport written to {args.output / 'report.html'}")


if __name__ == "__main__":
    main()
