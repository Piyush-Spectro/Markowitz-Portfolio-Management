"""Forecast terminal adjusted prices with training-only normalization.

The response is log(P_terminal / P_anchor). This scale-free price target is
mapped back to a strictly positive adjusted price, then used to rank stocks.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PricePrediction:
    prices: np.ndarray
    audit: dict
    checkpoint: dict | None = None


def price_windows(prices, lookback=20, horizon=21, execution_delay=1):
    p = np.asarray(prices, float)
    span = horizon + execution_delay
    if (p.ndim != 2 or not np.isfinite(p).all() or (p <= 0).any() or lookback < 2
            or horizon < 1 or execution_delay < 0 or len(p) < lookback + span + 40):
        raise ValueError('Need enough finite positive prices for chronological windows')
    anchors = np.arange(lookback - 1, len(p) - span)
    x = np.stack([np.log(p[t - lookback + 1:t + 1] / p[t]) for t in anchors])
    y = np.stack([np.log(p[t + span] / p[t]) for t in anchors])
    return x, y, anchors


def build_price_network(assets, architecture='rcnn_bilstm', hidden=16, recurrent_steps=3):
    import torch
    from torch import nn
    if architecture not in {'rcnn_bilstm', 'lstm', 'bilstm'}:
        raise ValueError('Unknown price network architecture')

    class RecurrentConvolution(nn.Module):
        """Shared-weight convolutional refinement within the observed window.

        h_0 = relu(W_input * x); h_j = relu(W_input * x + W_rec * h_(j-1)).
        The recurrent kernel is reused, rather than stacking unrelated CNN layers.
        """
        def __init__(self):
            super().__init__()
            self.input_conv = nn.Conv1d(assets, hidden, 3, padding=1)
            self.recurrent_conv = nn.Conv1d(hidden, hidden, 3, padding=1, bias=False)

        def forward(self, value):
            drive = self.input_conv(value.transpose(1, 2))
            state = torch.relu(drive)
            for _ in range(recurrent_steps):
                state = torch.relu(drive + self.recurrent_conv(state))
            return state.transpose(1, 2)

    class PriceNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.rcnn = RecurrentConvolution() if architecture == 'rcnn_bilstm' else None
            self.lstm = nn.LSTM(hidden if self.rcnn else assets, hidden, batch_first=True,
                                bidirectional=architecture != 'lstm')
            self.head = nn.Linear(hidden * (1 if architecture == 'lstm' else 2), assets)

        def forward(self, value):
            if self.rcnn is not None:
                value = self.rcnn(value)
            _, (hidden_state, _) = self.lstm(value)
            state = hidden_state[-1] if architecture == 'lstm' else torch.cat([hidden_state[-2], hidden_state[-1]], 1)
            return self.head(state)

    return PriceNetwork()


def predict_prices(prices, method='rcnn_bilstm', lookback=20, horizon=21, execution_delay=1,
                   epochs=20, seed=42, keep_checkpoint=False):
    p = np.asarray(prices, float)
    x, y, anchors = price_windows(p, lookback, horizon, execution_delay)
    span = horizon + execution_delay
    if method == 'historical':
        log_ratio = np.diff(np.log(p), axis=0).mean(axis=0) * span
        return PricePrediction(p[-1] * np.exp(log_ratio), {'method': method})
    if method == 'ridge':
        z = x.reshape(len(x), -1)
        center, scale = z.mean(0), np.maximum(z.std(0), 1e-6)
        z = (z - center) / scale
        # Dual solution avoids an asset_count*lookback square system.
        beta = z.T @ np.linalg.solve(z @ z.T + 100 * np.eye(len(z)), y - y.mean(0))
        latest = np.log(p[-lookback:] / p[-1]).reshape(-1)
        log_ratio = ((latest - center) / scale) @ beta + y.mean(0)
        return PricePrediction(p[-1] * np.exp(log_ratio), {'method': method})
    if method not in {'lstm', 'bilstm', 'rcnn_bilstm'} or epochs < 1:
        raise ValueError('Invalid price forecast method or epoch count')
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    split = int(len(x) * 0.8)
    train_end = split - span
    if train_end < 20 or len(x) - split < 5:
        raise ValueError('Insufficient data for purged price validation')
    center, scale = x[:train_end].mean((0, 1)), np.maximum(x[:train_end].std((0, 1)), 1e-6)
    target_center, target_scale = y[:train_end].mean(0), np.maximum(y[:train_end].std(0), 1e-6)
    xt = torch.tensor((x - center) / scale, dtype=torch.float32)
    yt = torch.tensor((y - target_center) / target_scale, dtype=torch.float32)
    model = build_price_network(p.shape[1], method)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
    loss_fn = torch.nn.MSELoss()
    best, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(xt[:train_end]), yt[:train_end])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xt[split:]), yt[split:]))
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError('Price network training produced no finite validation loss')
    model.load_state_dict(best_state)
    latest = np.log(p[-lookback:] / p[-1])
    with torch.no_grad():
        log_ratio = model(torch.tensor((latest - center) / scale, dtype=torch.float32)[None]).numpy()[0]
    prediction = p[-1] * np.exp(log_ratio * target_scale + target_center)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise RuntimeError('Price forecast is not finite and positive')
    checkpoint = None
    if keep_checkpoint:
        checkpoint = {'state_dict': best_state, 'architecture': method, 'assets': p.shape[1],
                      'lookback': lookback, 'horizon': horizon, 'execution_delay': execution_delay,
                      'center': center.tolist(), 'scale': scale.tolist(),
                      'target_center': target_center.tolist(), 'target_scale': target_scale.tolist()}
    audit = {'method': method, 'best_epoch': best_epoch, 'validation_mse_standardized': best,
             'training_windows': train_end, 'validation_windows': len(x) - split,
             'last_train_target_index': int(anchors[train_end - 1] + span),
             'first_validation_anchor_index': int(anchors[split])}
    return PricePrediction(prediction, audit, checkpoint)


def predict_from_checkpoint(checkpoint, observed_prices):
    """Replay a saved neural model without retraining (same asset order required)."""
    import torch
    p = np.asarray(observed_prices, float)
    if p.ndim != 2 or p.shape[1] != checkpoint['assets'] or len(p) < checkpoint['lookback']:
        raise ValueError('Checkpoint asset count or lookback does not match input')
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Observed prices must be finite and positive')
    model = build_price_network(checkpoint['assets'], checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    latest = np.log(p[-checkpoint['lookback']:] / p[-1])
    z = (latest - np.asarray(checkpoint['center'])) / np.asarray(checkpoint['scale'])
    with torch.no_grad():
        output = model(torch.tensor(z, dtype=torch.float32)[None]).numpy()[0]
    return p[-1] * np.exp(output * np.asarray(checkpoint['target_scale']) + np.asarray(checkpoint['target_center']))
