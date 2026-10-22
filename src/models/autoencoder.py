"""Task 7 - Spectral autoencoder anomaly detector (pure NumPy, no PyTorch needed).

    from src.models.autoencoder import SpectralAutoencoder
    ae = SpectralAutoencoder(latent_dim=8, epochs=40).fit(cube)
    scores = ae.score(cube)                 # mean squared reconstruction error per pixel
    per_band = ae.band_errors(cube[r, c])   # which bands it failed to reconstruct

The network squeezes each spectrum through a small bottleneck and rebuilds it:

    B bands -> hidden (tanh) -> latent -> hidden (tanh) -> B bands

Because almost every pixel is background, the network learns to rebuild
background spectra well; rare spectra (anomalies) rebuild poorly, so their
reconstruction error is high.

``trim_fraction`` makes training robust: after a warm-up, the worst-
reconstructed pixels in each batch are left out of the loss, so the network is
not taught to reproduce the anomalies we are trying to find.
"""
from __future__ import annotations

import time

import numpy as np

from src.utils.validation import HSIValidationError, check_float, check_int, validate_cube


class SpectralAutoencoder:
    def __init__(self, hidden_dim: int = 64, latent_dim: int = 8, epochs: int = 40, batch_size: int = 256,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, trim_fraction: float = 0.02,
                 warmup_epochs: int = 5, max_train_pixels: int = 50_000, patience: int = 6,
                 random_state: int = 0, verbose: bool = False):
        self.hidden_dim = check_int(hidden_dim, "hidden_dim", 2, 4096)
        self.latent_dim = check_int(latent_dim, "latent_dim", 1, 1024)
        self.epochs = check_int(epochs, "epochs", 1, 10_000)
        self.batch_size = check_int(batch_size, "batch_size", 1)
        self.learning_rate = check_float(learning_rate, "learning_rate", 0, 1, lo_inclusive=False)
        self.weight_decay = check_float(weight_decay, "weight_decay", 0)
        self.trim_fraction = check_float(trim_fraction, "trim_fraction", 0, 0.5, hi_inclusive=False)
        self.warmup_epochs = check_int(warmup_epochs, "warmup_epochs", 0)
        self.max_train_pixels = check_int(max_train_pixels, "max_train_pixels", 100)
        self.patience = check_int(patience, "patience", 1)
        self.random_state = random_state
        self.verbose = verbose
        if self.latent_dim >= self.hidden_dim:
            raise HSIValidationError("latent_dim must be smaller than hidden_dim for a bottleneck.")

    # ---------------------------------------------------------------- network
    def _init(self, b, rng):
        dims = [b, self.hidden_dim, self.latent_dim, self.hidden_dim, b]
        self.W_ = [rng.normal(0, np.sqrt(1.0 / i), (i, o)) for i, o in zip(dims[:-1], dims[1:])]
        self.b_ = [np.zeros(o) for o in dims[1:]]
        self._m = [np.zeros_like(p) for p in self.W_ + self.b_]
        self._v = [np.zeros_like(p) for p in self.W_ + self.b_]
        self._t = 0

    def _forward(self, x):
        acts = [x]
        h = x
        for i, (W, bias) in enumerate(zip(self.W_, self.b_)):
            h = h @ W + bias
            if i < len(self.W_) - 1 and i != 1:   # tanh on hidden layers, linear latent and output
                h = np.tanh(h)
            acts.append(h)
        return acts

    def _backward(self, acts, grad_out):
        gW, gb = [None] * 4, [None] * 4
        g = grad_out
        for i in reversed(range(4)):
            gW[i] = acts[i].T @ g + self.weight_decay * self.W_[i]
            gb[i] = g.sum(0)
            if i > 0:
                g = g @ self.W_[i].T
                if i - 1 != 1:                      # layer i-1 output used tanh
                    g = g * (1 - acts[i] ** 2)
        return gW + gb

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        params = self.W_ + self.b_
        for p, g, m, v in zip(params, grads, self._m, self._v):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            p -= self.learning_rate * (m / (1 - b1 ** self._t)) / (np.sqrt(v / (1 - b2 ** self._t)) + eps)

    # ----------------------------------------------------------------- public
    def fit(self, cube, mask=None):
        cube = validate_cube(cube)
        x = cube.reshape(-1, cube.shape[2]).astype(np.float64)
        if mask is not None:
            x = x[np.asarray(mask, bool).ravel()]
        if not np.isfinite(x).all():
            raise HSIValidationError("Autoencoder input contains NaN/inf. Run preprocess() first.")
        rng = np.random.default_rng(self.random_state)
        if x.shape[0] > self.max_train_pixels:
            x = x[rng.choice(x.shape[0], self.max_train_pixels, replace=False)]
        self.mu_, self.sd_ = x.mean(0), x.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        x = (x - self.mu_) / self.sd_
        self._init(x.shape[1], rng)

        self.history_, best, stale = [], np.inf, 0
        t0 = time.perf_counter()
        bs = min(self.batch_size, x.shape[0])
        for epoch in range(self.epochs):
            order = rng.permutation(x.shape[0])
            total = 0.0
            for i in range(0, x.shape[0], bs):
                xb = x[order[i:i + bs]]
                acts = self._forward(xb)
                err = acts[-1] - xb
                per = (err ** 2).mean(1)
                weight = np.ones(len(xb))
                if epoch >= self.warmup_epochs and self.trim_fraction > 0 and len(xb) > 10:
                    weight[per > np.quantile(per, 1 - self.trim_fraction)] = 0.0
                g = (2.0 / xb.shape[1]) * err * weight[:, None] / max(weight.sum(), 1)
                self._adam(self._backward(acts, g))
                total += float((per * weight).sum() / max(weight.sum(), 1)) * len(xb)
            loss = total / x.shape[0]
            if not np.isfinite(loss):
                raise HSIValidationError("Autoencoder training diverged (loss is NaN). Lower learning_rate.")
            self.history_.append(loss)
            if self.verbose:
                print(f"epoch {epoch + 1:3d}  loss {loss:.5f}")
            if loss < best * 0.999:
                best, stale = loss, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.train_seconds_ = time.perf_counter() - t0
        self.epochs_run_ = len(self.history_)
        return self

    def reconstruct(self, cube):
        """Reconstructed spectra in the original (input) units, same shape as ``cube``."""
        self._check()
        arr = np.asarray(cube, dtype=np.float64)
        shape = arr.shape
        x = (arr.reshape(-1, shape[-1]) - self.mu_) / self.sd_
        out = np.empty_like(x)
        for i in range(0, len(x), 100_000):
            out[i:i + 100_000] = self._forward(x[i:i + 100_000])[-1]
        return (out * self.sd_ + self.mu_).reshape(shape)

    def band_errors(self, cube):
        """Squared reconstruction error per band (standardised units). Works on a pixel or a cube."""
        self._check()
        arr = np.asarray(cube, dtype=np.float64)
        if arr.shape[-1] != self.mu_.size:
            raise HSIValidationError(f"Autoencoder was trained on {self.mu_.size} bands but got {arr.shape[-1]}.")
        x = (arr.reshape(-1, arr.shape[-1]) - self.mu_) / self.sd_
        err = np.empty_like(x)
        for i in range(0, len(x), 100_000):
            err[i:i + 100_000] = (self._forward(x[i:i + 100_000])[-1] - x[i:i + 100_000]) ** 2
        return err.reshape(arr.shape)

    def score(self, cube):
        """Mean squared reconstruction error per pixel, shape (H, W)."""
        cube = validate_cube(cube, min_pixels=1)
        return self.band_errors(cube).mean(-1)

    def fit_score(self, cube, mask=None):
        return self.fit(cube, mask).score(cube)

    def _check(self):
        if not hasattr(self, "W_"):
            raise HSIValidationError("Autoencoder must be fitted first.")
