"""
LSTM / GRU classifiers of Goel et al. (2021), with a scikit-learn-like interface.

  o2o (one-to-one) : the input sequence is the current ball only (length 1)
  m2o (many-to-one): the input sequence is the last `window` balls of the same innings,
                     ending at the current ball (zero-padded at the start of the innings)

  flavours: vanilla (1 layer), stacked (2 layers), bidirec (bidirectional, 1 layer).
  The bidirectional model reads only the past window (never future balls), so it does not leak.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


def make_windows(X: np.ndarray, match_ids: np.ndarray, window: int) -> np.ndarray:
    """X rows must be ordered by ball inside each match. Returns [N, window, F]."""
    n, f = X.shape
    out = np.zeros((n, window, f), dtype=np.float32)
    start = 0
    for i in range(1, n + 1):
        if i == n or match_ids[i] != match_ids[start]:
            for j in range(start, i):
                lo = max(start, j - window + 1)
                seg = X[lo:j + 1]
                out[j, window - len(seg):] = seg
            start = i
    return out


class _Net(nn.Module):
    def __init__(self, n_feat, cell, layers, bidir, hidden):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_feat, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        self.head = nn.Linear(hidden * (2 if bidir else 1), 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class SeqClassifier:
    def __init__(self, cell="lstm", flavour="vanilla", seq="o2o", window=6, hidden=32,
                 epochs=8, batch=512, lr=2e-3, seed=42):
        self.cell, self.flavour, self.seq = cell, flavour, seq
        self.window = 1 if seq == "o2o" else window
        self.hidden, self.epochs, self.batch, self.lr, self.seed = hidden, epochs, batch, lr, seed

    def _prep(self, X, groups, fit=False):
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        return make_windows(X, np.asarray(groups), self.window)

    def fit(self, X, y, groups):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        W = torch.from_numpy(self._prep(X, groups, fit=True))
        Y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        layers = 2 if self.flavour == "stacked" else 1
        self.net = _Net(W.shape[2], self.cell, layers, self.flavour == "bidirec", self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(W)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(W[idx]), Y[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X, groups):
        W = torch.from_numpy(self._prep(X, groups))
        self.net.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(self.net(W[s:s + 4096])) for s in range(0, len(W), 4096)])
        p = p.numpy()
        return np.c_[1 - p, p]


# The 10 deep-learning variants reported in Table 3 of the paper
DL_VARIANTS = {
    "o2o MF bidirec GRU": dict(cell="gru", flavour="bidirec", seq="o2o"),
    "o2o MF Stacked GRU": dict(cell="gru", flavour="stacked", seq="o2o"),
    "o2o MF vanilla GRU": dict(cell="gru", flavour="vanilla", seq="o2o"),
    "o2o MF bidirec lstm": dict(cell="lstm", flavour="bidirec", seq="o2o"),
    "o2o MF Stacked lstm": dict(cell="lstm", flavour="stacked", seq="o2o"),
    "o2o MF vanilla lstm": dict(cell="lstm", flavour="vanilla", seq="o2o"),
    "m2o MF Stacked GRU": dict(cell="gru", flavour="stacked", seq="m2o"),
    "m2o MF vanilla GRU": dict(cell="gru", flavour="vanilla", seq="m2o"),
    "m2o MF Stacked LSTM": dict(cell="lstm", flavour="stacked", seq="m2o"),
    "m2o MF vanilla LSTM": dict(cell="lstm", flavour="vanilla", seq="m2o"),
}
