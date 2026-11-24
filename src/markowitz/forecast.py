"""Forecast the mean daily return over the next holding period."""
import numpy as np


def supervised_windows(returns, lookback=20, horizon=21):
    x = np.asarray(returns, float)
    if (x.ndim != 2 or not np.isfinite(x).all() or lookback < 2 or horizon < 1
            or len(x) < lookback + horizon + 20):
        raise ValueError("Insufficient or invalid observations for forecast windows.")
    ends = range(lookback, len(x) - horizon + 1)
    features = np.stack([x[t - lookback:t] for t in ends])
    labels = np.stack([x[t:t + horizon].mean(axis=0) for t in ends])
    return features, labels


def ridge_forecast(returns, lookback=20, horizon=21, penalty=100.0):
    """Multi-output ridge with train-only scaling and an unpenalized intercept."""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Ridge penalty must be positive.")
    features, labels = supervised_windows(returns, lookback, horizon)
    x = features.reshape(len(features), -1)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    z = (x - center) / scale
    label_mean = labels.mean(axis=0)
    coef = np.linalg.solve(z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (labels - label_mean))
    latest = np.asarray(returns)[-lookback:].reshape(-1)
    return ((latest - center) / scale) @ coef + label_mean


def neural_forecast(returns, lookback=20, horizon=21, epochs=12, seed=42, architecture="cnn_bilstm"):
    """Small CPU model with purged chronological validation and best-epoch restore.

    Bidirectionality is confined to the observed input window. No future labels
    or held-out backtest returns are included in that window.
    """
    import torch
    from torch import nn
    if epochs < 1 or architecture not in {"cnn_bilstm", "lstm", "bilstm"}:
        raise ValueError("Invalid neural training configuration.")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y = supervised_windows(returns, lookback, horizon)
    split = int(len(x) * 0.8)
    train_end = split - horizon + 1  # Purge labels that overlap validation labels.
    if train_end < 20 or len(x) - split < 5:
        raise ValueError("Not enough data for purged neural validation.")
    center = x[:train_end].mean(axis=(0, 1))
    scale = np.maximum(x[:train_end].std(axis=(0, 1)), 1e-6)
    target_center = y[:train_end].mean(axis=0)
    target_scale = np.maximum(y[:train_end].std(axis=0), 1e-6)
    xt = torch.tensor((x - center) / scale, dtype=torch.float32)
    yt = torch.tensor((y - target_center) / target_scale, dtype=torch.float32)
    n = x.shape[2]

    class ReturnNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(n, 16, kernel_size=3, padding=1) if architecture == "cnn_bilstm" else None
            self.rnn = nn.LSTM(16 if self.conv else n, 16, batch_first=True,
                               bidirectional=architecture != "lstm")
            self.head = nn.Linear(32 if architecture != "lstm" else 16, n)

        def forward(self, value):
            if self.conv is not None:
                value = torch.relu(self.conv(value.transpose(1, 2))).transpose(1, 2)
            _, (hidden, _) = self.rnn(value)
            state = torch.cat([hidden[-2], hidden[-1]], dim=1) if architecture != "lstm" else hidden[-1]
            return self.head(state)

    model = ReturnNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
    loss_fn = nn.MSELoss()
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(xt[:train_end]), yt[:train_end])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xt[split:]), yt[split:]))
        if val < best:
            best, best_state = val, {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Neural training produced no finite validation loss.")
    model.load_state_dict(best_state)
    latest = torch.tensor((np.asarray(returns)[-lookback:] - center) / scale, dtype=torch.float32)[None]
    with torch.no_grad():
        prediction = model(latest).numpy()[0] * target_scale + target_center
    return prediction


def forecast(returns, method="ridge", horizon=21, epochs=12, seed=42):
    if method == "historical":
        result = np.asarray(returns).mean(axis=0)
    elif method == "ridge":
        result = ridge_forecast(returns, horizon=horizon)
    elif method in {"cnn_bilstm", "lstm", "bilstm"}:
        result = neural_forecast(returns, horizon=horizon, epochs=epochs, seed=seed, architecture=method)
    else:
        raise ValueError(f"Unknown forecast method: {method}")
    if not np.isfinite(result).all():
        raise RuntimeError("Forecast contains non-finite values.")
    return result
