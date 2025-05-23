# cox_nnet_pytorch.py
"""
PyTorch re‑implementation of the original Theano Cox‑nnet script.

Key differences
---------------
* Uses `torch.nn.Module` subclasses – **no Theano shared variables**.
* Training loop relies on the built–in torch `optim` package (SGD / Nesterov).
* Model serialization with `torch.save` / `torch.load`.
* NumPy remains for data wrangling, scikit‑learn for CV utilities – exactly as in the
  reference implementation.

The public API purposefully mirrors the original:
    - ``train_cox_mlp``  – fits a network and returns it.
    - ``predict``        – predicts risk scores (theta) for new data.
    - ``var_importance`` – Fischer (2015) permutation score.

"""

from __future__ import annotations
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

TORCH_DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x: np.ndarray | torch.Tensor, *, dtype=TORCH_DTYPE) -> torch.Tensor:
    """Convenience – converts ``x`` to a contiguous Torch tensor on the default device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


# -----------------------------------------------------------------------------
#  Model building blocks
# -----------------------------------------------------------------------------

class CoxRegression(nn.Module):
    """Final linear layer producing *theta* (risk score).

    No bias term – identical to the Theano code.
    """

    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, 1)
        return self.linear(x).squeeze(-1)  # (batch,)


class HiddenLayer(nn.Module):
    """Single hidden layer that may consume multiple *input slices* (same as `map`)."""

    def __init__(self, in_features: int, out_features: int, activation=nn.Tanh,dropout=0.5):
        super().__init__()
        # Xavier/Glorot – equivalent to the uniform init in the original script
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.linear(x.double())))  # (batch, out_features)


class CoxMLP(nn.Module):
    """End‑to‑end network – multiple hidden blocks → CoxRegression."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None,dropout: float = 0.5):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [int(math.ceil(math.sqrt(input_dim)))]

        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers.append(HiddenLayer(last, h, dropout=dropout))
            #append tanh and dropout
            last = h
        self.hidden = nn.Sequential(*layers)
        self.cox = CoxRegression(last)

    # ------------------------------------------------------------------
    #  Forward + special losses
    # ------------------------------------------------------------------
    def forward(self, x):
        h = self.hidden(x)
        return self.cox(h)  # theta (batch,)

    @staticmethod
    def neg_log_partial_lik(theta: torch.Tensor, y_time: torch.Tensor, y_event: torch.Tensor) -> torch.Tensor:
        """Vectorised Breslow partial likelihood (negative)."""
        # R_i = patients with time >= t_i  → build risk set via broadcasting.
        # Works on *batch* but must be full dataset for correct likelihood.
        exp_theta = torch.exp(theta)
        # (n, n) bool mask – True if j in risk set of i
        risk_mat = (y_time.unsqueeze(0) <= y_time.unsqueeze(1)).float()
        log_cumsum = torch.log((exp_theta.unsqueeze(0) * risk_mat).sum(1))
        ll = (theta - log_cumsum) * y_event
        return -ll.mean()

    # convenience ----------------------------------------------------------------

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(to_tensor(x)).cpu().numpy()


# -----------------------------------------------------------------------------
#  Training / evaluation helpers
# -----------------------------------------------------------------------------

def train_cox_mlp(
    x_train: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
    x_val: np.ndarray | None = None,
    y_time_val: np.ndarray | None = None,
    y_event_val: np.ndarray | None = None,
    l2_reg: float = math.exp(-1),
    hidden_dims: list[int] | None = None,
    method: str = "nesterov",  # "momentum"|"sgd"
    lr: float = 1e-2,
    momentum: float = 0.9,
    lr_decay: float = 0.9,
    lr_growth: float = 1.0,
    eval_step: int = 23,
    max_iter: int = 10_000,
    stop_threshold: float = 0.995,
    patience: int = 2_000,
    dropout: float = 0.5,
    patience_incr: int = 2,
    seed: int = 123,
    verbose: bool | int = False,
    early_stopping: bool = True,
) -> CoxMLP:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    n_samples, n_features = x_train.shape
    x_t = to_tensor(x_train)
    t_t = to_tensor(y_time)
    e_t = to_tensor(y_event)

    model = CoxMLP(n_features, hidden_dims, dropout=dropout).to(device)

    nesterov = method == "nesterov"
    mom = momentum if method in {"momentum", "nesterov"} else 0.0
    optim = SGD(model.parameters(), lr=lr, momentum=mom, nesterov=nesterov, weight_decay=l2_reg)

    best_loss = float("inf")
    start = time.time()

    for it in range(max_iter):
        model.train()
        optim.zero_grad()
        theta = model(x_t)
        loss = CoxMLP.neg_log_partial_lik(theta, t_t, e_t)
        loss.backward()
        optim.step()

        # eval val loss
        if x_val is not None and y_time_val is not None and y_event_val is not None:
            model.eval()
            with torch.no_grad():
                val_theta = model(to_tensor(x_val))
                val_loss = CoxMLP.neg_log_partial_lik(val_theta, to_tensor(y_time_val), to_tensor(y_event_val))
                if verbose:
                    print(f"iter={it:5d} val_loss={val_loss:.6f}")

        # evaluation / LR adaptation --------------------------------------
        if (it % eval_step) == 0:
            if x_val is not None and y_time_val is not None and y_event_val is not None:
                # validation loss
                cur = val_loss.item()
            else:
                # training loss
                cur = loss.item()
            if cur < best_loss * stop_threshold:
                best_loss = cur
                patience = max(patience, it * patience_incr)
                if verbose:
                    print(f"iter={it:5d} loss={cur:.6f}")
                # optionally grow LR
                for g in optim.param_groups:
                    g["lr"] *= lr_growth
            else:
                # decay LR
                for g in optim.param_groups:
                    g["lr"] *= lr_decay

        if it >= patience and early_stopping:
            break

    if verbose:
        dur = time.time() - start
        print(f"Training finished: {it} iterations – {dur:.1f}s (best loss {best_loss:.5f})")

    return model


# -----------------------------------------------------------------------------
#  Cross‑validation, search, utilities (port of the Theano helpers)
# -----------------------------------------------------------------------------

def c_index(model: CoxMLP, x: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    theta = model.predict(x)
    concord, permissible = 0.0, 0.0
    n = len(t)
    for i in range(n):
        if e[i] != 1:
            continue
        mask = t > t[i]
        permissible += mask.sum()
        concord += ((theta[mask] < theta[i]).sum() + 0.5 * (theta[mask] == theta[i]).sum())
    return float(concord / permissible)


def cv_loglikelihood(model: CoxMLP, x_full, t_full, e_full, x_train, t_train, e_train):
    # full
    theta_full = model.predict(x_full)
    exp_theta = np.exp(theta_full)
    risk = (t_full.reshape(-1, 1) <= t_full.reshape(1, -1)).astype(float)
    pl_full = ((theta_full - np.log((exp_theta * risk).sum(1))) * e_full).sum()

    # train subset
    theta_tr = model.predict(x_train)
    exp_tr = np.exp(theta_tr)
    risk_tr = (t_train.reshape(-1, 1) <= t_train.reshape(1, -1)).astype(float)
    pl_tr = ((theta_tr - np.log((exp_tr * risk_tr).sum(1))) * e_train).sum()
    return pl_full - pl_tr


# -----------------------------------------------------------------------------
#  Variable importance (Fischer 2015) – mean effect when permuting a single var
# -----------------------------------------------------------------------------

def var_importance(model: CoxMLP, x_train, t_train, e_train):
    n, p = x_train.shape
    risk_full = CoxMLP.neg_log_partial_lik(torch.tensor(model.predict(x_train)),
                                           to_tensor(t_train),
                                           to_tensor(e_train))
    imp = np.zeros(p)
    for j in range(p):
        x_mod = x_train.copy()
        x_mod[:, j] = x_mod[:, j].mean()
        imp[j] = (
            CoxMLP.neg_log_partial_lik(torch.tensor(model.predict(x_mod)),
                                        to_tensor(t_train),
                                        to_tensor(e_train))
            - risk_full
        ).item()
    return imp


# -----------------------------------------------------------------------------
#  Persistence helpers
# -----------------------------------------------------------------------------

def save_model(model: CoxMLP, path: str | Path):
    torch.save(model.state_dict(), Path(path))


def load_model(path: str | Path, input_dim: int, hidden_dims=None) -> CoxMLP:
    model = CoxMLP(input_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

class CoxPartialLikelihoodLoss(nn.Module):
    """
    Negative Breslow partial log-likelihood for Cox models.

    Parameters
    ----------
    reduction : {'mean', 'sum', 'none'}, default='mean'
        • 'mean' – average over events (original behaviour)
        • 'sum'  – sum over events
        • 'none' – return the per-instance contributions (useful for debugging)
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        self.reduction = reduction

    @staticmethod
    def _neg_ll(theta: torch.Tensor,
                y_labels: torch.Tensor
                ) -> torch.Tensor:
        """
        Implements

            − 1/N ∑_i e_i [ θ_i − log Σ_{j∈R_i} exp θ_j ],

        where R_i = { j | t_j ≥ t_i }.
        """
        y_time, y_event = y_labels[:, 0], y_labels[:, 1]
        exp_theta = torch.exp(theta)                              # (n,)
        # risk set mask (n × n): 1 if t_j ≥ t_i
        risk_mat  = (y_time.unsqueeze(0) <= y_time.unsqueeze(1)).float()
        denom     = (exp_theta.unsqueeze(0) * risk_mat).sum(1)    # (n,)
        log_cumsum = torch.log(denom + 1e-9)                      # ε for stability
        ll_vec    = (theta - log_cumsum) * y_event               # (n,)
        return -ll_vec  # negative log-lik

    # ------------------------------------------------------------------ #

    def forward(self,
                theta:   torch.Tensor,
                y_labels:  torch.Tensor
                ) -> torch.Tensor:
        """
        All three tensors must have identical first dimension `N`.

        * `theta`   – risk scores predicted by the model, shape (N,)
        y_labels, composed of:
        * `y_time`  – observed times                 , shape (N,)
        * `y_event` – event indicators (1=event, 0=censored), shape (N,)
        """
        nll_vec = self._neg_ll(theta, y_labels)

        if   self.reduction == "sum":
            return nll_vec.sum()
        elif self.reduction == "mean":
            return nll_vec.mean()                # matches original function
        else:                                     # 'none'
            return nll_vec



# -----------------------------------------------------------------------------
#  Example usage (will not run on import)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy synthetic data --------------------------------------------------
    rng = np.random.default_rng(0)
    n, p = 512, 100
    x = rng.standard_normal((n, p)).astype(np.float32)
    beta = rng.standard_normal(p)
    linpred = x.dot(beta)
    t = rng.exponential(scale=np.exp(-linpred))
    e = rng.integers(0, 2, size=n)

    model = train_cox_mlp(x, t, e, hidden_dims=[32, 16], verbose=True)
    print("C‑index on train:", c_index(model, x, t, e))
