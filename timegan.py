"""
Conditional TimeGAN for SCG signal augmentation.

Implements Time-series Generative Adversarial Network (Yoon et al. 2019)
with class conditioning, designed for 3-channel SCG segments (800 samples).

Architecture:
  Embedder   : GRU  → MLP   (real seq → latent space)
  Recovery   : GRU  → MLP   (latent → reconstructed seq)
  Generator  : GRU  → MLP   (noise + class → synthetic latent)
  Discriminator : GRU → MLP (latent + class → real/fake)

Loss components (Yoon et al. 2019):
  1) Reconstruction loss  — MSE(real, recovered)
  2) Unsupervised loss    — BCE from discriminator (real vs fake latent)
  3) Supervised loss      — MSE(next-step embedding prediction)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _one_hot(labels, num_classes, device):
    """Convert integer labels to one-hot vectors."""
    if labels.dim() == 1:
        return F.one_hot(labels, num_classes=num_classes).float().to(device)
    return labels.float().to(device)


# ──────────────────────────────────────────────
#  Sub-modules
# ──────────────────────────────────────────────

class _GRU_Encoder(nn.Module):
    """Single-layer GRU -> linear projection."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)  # (B, T, H)
        return self.proj(out)


class _GRU_Discriminator(nn.Module):
    """GRU -> binary classifier with class conditioning."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim + num_classes,  # conditioned on class
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, class_labels):
        # x: (B, T, D), class_labels: (B, C)
        cond = class_labels.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, C)
        x_cond = torch.cat([x, cond], dim=-1)  # (B, T, D + C)
        out, _ = self.gru(x_cond)  # (B, T, H)
        # Use last timestep for classification
        return self.classifier(out[:, -1, :])  # (B, 1)


# ──────────────────────────────────────────────
#  Main TimeGAN model
# ──────────────────────────────────────────────

class TimeGAN(nn.Module):
    """
    Conditional Time-Series Generative Adversarial Network.

    Parameters
    ----------
    feature_dim : int
        Number of input channels (3 for AccX, AccY, AccZ).
    seq_len : int
        Length of each time series (800 samples).
    latent_dim : int
        Dimensionality of the latent embedding space.
    hidden_dim : int
        Hidden size for all GRU modules.
    num_classes : int
        Number of conditioning classes (AS, MR, MS, AR, N, etc.).
    num_layers : int
        Number of GRU layers in each sub-network.
    device : torch.device
        Device to place the model on.
    """

    def __init__(
        self,
        feature_dim=3,
        seq_len=800,
        latent_dim=32,
        hidden_dim=64,
        num_classes=5,
        num_layers=1,          # 1 layer is much faster on MPS/CPU
        device=None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")

        # ── Embedder & Recovery (autoencoder) ──
        self.embedder = _GRU_Encoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_layers,
        )
        self.recovery = _GRU_Encoder(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            num_layers=num_layers,
        )

        # ── Generator & Discriminator ──
        # Generator input: noise_z (latent_dim) + class label (num_classes)
        self.generator = _GRU_Encoder(
            input_dim=latent_dim + num_classes,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_layers,
        )
        self.discriminator = _GRU_Discriminator(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
        )

        # ── Supervised next-step predictor (in latent space) ──
        self.supervisor = nn.Linear(latent_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def to(self, device):
        self.device = device
        return super().to(device)

    # ── Forward passes ──

    def embed(self, X):
        """Map real sequence to latent space."""
        return self.embedder(X)

    def recover(self, H):
        """Reconstruct sequence from latent space."""
        return self.recovery(H)

    def generate(self, Z, class_labels):
        """Generate latent sequence from noise + class conditioning."""
        # Z: (B, T, latent_dim), class_labels: (B, C)
        cond = class_labels.unsqueeze(1).expand(-1, Z.size(1), -1)  # (B, T, C)
        z_cond = torch.cat([Z, cond], dim=-1)  # (B, T, latent_dim + C)
        return self.generator(z_cond)

    def discriminate(self, H, class_labels):
        """Classify latent sequence as real (1) or fake (0)."""
        return self.discriminator(H, class_labels)

    def supervise(self, H):
        """Predict next timestep embedding."""
        return self.supervisor(H)

    # ── Full autoencode ──

    def autoencode(self, X):
        """Encode then decode."""
        H = self.embed(X)
        X_hat = self.recover(H)
        return X_hat, H

    # ── Training helpers ──

    @staticmethod
    def _random_times(B, T, latent_dim, device):
        return torch.randn(B, T, latent_dim, device=device)


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────

def train_timegan(
    model,
    train_data,      # numpy array (N, 3, 800) or (N, 800, 3)
    train_labels,    # numpy array (N,)  integer class labels
    num_epochs=200,
    batch_size=64,
    lr=1e-3,
    lambda_rec=1.0,
    lambda_sup=1.0,
    lambda_gp=10.0,  # gradient penalty weight
    log_callback=None,
    device=None,
):
    """
    Train the conditional TimeGAN on real SCG segments.

    Parameters
    ----------
    model : TimeGAN
    train_data : np.ndarray
        Shape (N, 3, 800) or (N, 800, 3) — segments.
    train_labels : np.ndarray
        Shape (N,) — integer class indices.
    num_epochs : int
    batch_size : int
    lr : float
    lambda_rec, lambda_sup, lambda_gp : float
        Loss weighting factors.
    log_callback : callable or None
        Called with (epoch, total, message) for logging.
    device : torch.device or None

    Returns
    -------
    model : TimeGAN (trained)
    losses : dict of lists
    """
    device = device or model.device
    # Ensure data is (N, seq_len, feature_dim)
    if train_data.ndim == 3 and train_data.shape[1] == model.feature_dim and train_data.shape[2] == model.seq_len:
        # (N, 3, 800) -> (N, 800, 3)
        data = np.transpose(train_data, (0, 2, 1)).astype(np.float32)
    elif train_data.ndim == 3 and train_data.shape[2] == model.feature_dim:
        data = train_data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected data shape: {train_data.shape}")

    labels = np.asarray(train_labels, dtype=np.int64)
    N = len(data)

    model = model.to(device)

    # Optimizers
    opt_ae = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=lr,
    )
    opt_gs = torch.optim.Adam(
        list(model.generator.parameters())
        + list(model.supervisor.parameters()),
        lr=lr,
    )
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    losses = {"rec": [], "supervised": [], "d_real": [], "d_fake": [], "g_adv": []}

    def _log(epoch, msg):
        if log_callback:
            log_callback(epoch, num_epochs, msg)

    # ── Phase 1: Embedding pre-training ──
    _log(0, "[TimeGAN] Phase 1: Embedding pre-training...")
    for epoch in range(num_epochs // 4):
        perm = np.random.permutation(N)
        epoch_rec_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            X = torch.tensor(data[idx], device=device)
            X_hat, _ = model.autoencode(X)
            loss = mse_loss(X_hat, X)

            opt_ae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_ae.step()

            epoch_rec_loss += loss.item()
            n_batches += 1

        avg_rec = epoch_rec_loss / max(n_batches, 1)
        losses["rec"].append(avg_rec)
        _log(epoch + 1, f"  Embedding epoch {epoch+1}/{num_epochs//4} | Rec Loss: {avg_rec:.6f}")

    # ── Phase 2: Joint adversarial training ──
    _log(0, "[TimeGAN] Phase 2: Joint adversarial training...")
    for epoch in range(num_epochs):
        perm = np.random.permutation(N)
        epoch_rec_loss = 0.0
        epoch_sup_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        epoch_g_adv = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            X = torch.tensor(data[idx], device=device)            # (B, T, F)
            y = torch.tensor(labels[idx], device=device)
            y_onehot = _one_hot(y, model.num_classes, device)
            B = X.size(0)

            # ── Embed real ──
            H_real = model.embed(X)                                # (B, T, L)

            # ── Generate synthetic latent ──
            Z = model._random_times(B, model.seq_len, model.latent_dim, device)
            H_fake = model.generate(Z, y_onehot)                  # (B, T, L)

            # ── Supervised next-step loss (in latent space) ──
            H_sup = model.supervise(H_real[:, :-1, :])            # (B, T-1, L)
            loss_sup = mse_loss(H_sup, H_real[:, 1:, :])

            # ── Reconstruction loss ──
            X_hat = model.recover(H_real)
            loss_rec = mse_loss(X_hat, X)

            # ── Discriminator loss ──
            d_real_logits = model.discriminate(H_real.detach(), y_onehot)  # (B, 1)
            d_fake_logits = model.discriminate(H_fake.detach(), y_onehot)  # (B, 1)

            d_real_labels = torch.ones_like(d_real_logits, device=device) * 0.9  # label smoothing
            d_fake_labels = torch.zeros_like(d_fake_logits, device=device)

            loss_d_real = bce_loss(d_real_logits, d_real_labels)
            loss_d_fake = bce_loss(d_fake_logits, d_fake_labels)
            loss_d = loss_d_real + loss_d_fake

            # ── Generator adversarial loss ──
            g_fake_logits = model.discriminate(H_fake, y_onehot)
            loss_g_adv = bce_loss(g_fake_logits, torch.ones_like(g_fake_logits, device=device))

            # ── Compute all losses ──
            loss_ae = loss_rec + 0.5 * loss_sup
            loss_gs = loss_g_adv + lambda_sup * loss_sup

            # ── Backward passes FIRST (all grads computed from same param version) ──
            opt_ae.zero_grad()
            loss_ae.backward(retain_graph=True)

            opt_gs.zero_grad()
            loss_gs.backward(retain_graph=True)

            opt_d.zero_grad()
            loss_d.backward()

            # ── Optimizer steps AFTER (in-place param updates don't affect gradients) ──
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_ae.step()
            opt_gs.step()
            opt_d.step()

            epoch_rec_loss += loss_rec.item()
            epoch_sup_loss += loss_sup.item()
            epoch_d_real += loss_d_real.item()
            epoch_d_fake += loss_d_fake.item()
            epoch_g_adv += loss_g_adv.item()
            n_batches += 1

        n = max(n_batches, 1)
        losses["rec"].append(epoch_rec_loss / n)
        losses["supervised"].append(epoch_sup_loss / n)
        losses["d_real"].append(epoch_d_real / n)
        losses["d_fake"].append(epoch_d_fake / n)
        losses["g_adv"].append(epoch_g_adv / n)

        _log(
            epoch + 1,
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"Rec: {losses['rec'][-1]:.6f} | "
            f"Sup: {losses['supervised'][-1]:.6f} | "
            f"D_real: {losses['d_real'][-1]:.4f} | "
            f"D_fake: {losses['d_fake'][-1]:.4f} | "
            f"G_adv: {losses['g_adv'][-1]:.4f}",
        )

    _log(0, "[TimeGAN] Training complete.")
    return model, losses


# ──────────────────────────────────────────────
#  Sample generation
# ──────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, num_samples, class_idx, num_classes=None):
    """
    Generate synthetic SCG segments for a given class.

    Parameters
    ----------
    model : TimeGAN
    num_samples : int
        Number of synthetic segments to generate.
    class_idx : int
        Target class index (0 .. num_classes-1).
    num_classes : int, optional
        Total number of classes. If None, uses model.num_classes.

    Returns
    -------
    synthetic : torch.Tensor
        Shape (num_samples, 3, 800) — matches the format used by TrainingWorker.
        All NaN/Inf values are replaced with 0, and values are clamped to [-5, 5]
        to prevent MPS segfaults from out-of-range data.
    """
    device = model.device
    model.eval()

    if num_classes is None:
        num_classes = model.num_classes

    B = min(num_samples, 256)  # generate in chunks to avoid OOM
    all_segments = []

    remaining = num_samples
    while remaining > 0:
        batch = min(B, remaining)
        Z = torch.randn(batch, model.seq_len, model.latent_dim, device=device)
        y = torch.full((batch,), class_idx, dtype=torch.long, device=device)
        y_onehot = _one_hot(y, num_classes, device)

        H_fake = model.generate(Z, y_onehot)
        X_fake = model.recover(H_fake)  # (B, 800, 3)

        # Convert to (B, 3, 800) format
        X_fake = X_fake.permute(0, 2, 1)  # (B, 3, 800)
        
        # Clamp and sanitize to prevent MPS segfaults from NaN/Inf
        X_fake = torch.nan_to_num(X_fake, nan=0.0, posinf=5.0, neginf=-5.0)
        X_fake = X_fake.clamp(-5.0, 5.0)
        
        all_segments.append(X_fake.cpu())
        remaining -= batch

    if not all_segments:
        return torch.empty((0, 3, model.seq_len))

    return torch.cat(all_segments, dim=0)[:num_samples]


# ──────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────

def save_timegan_checkpoint(model, path, losses=None):
    """
    Save a TimeGAN model checkpoint.

    Parameters
    ----------
    model : TimeGAN
    path : str
    losses : dict or None
        Training loss history to save alongside weights.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "feature_dim": model.feature_dim,
        "seq_len": model.seq_len,
        "latent_dim": model.latent_dim,
        "hidden_dim": model.hidden_dim,
        "num_classes": model.num_classes,
        "num_layers": model.num_layers,
        "losses": losses,
    }
    torch.save(state, path)


def load_timegan_checkpoint(path, device=None):
    """
    Load a TimeGAN model from a checkpoint file.

    Parameters
    ----------
    path : str
    device : torch.device or None

    Returns
    -------
    model : TimeGAN
    losses : dict or None
    """
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = TimeGAN(
        feature_dim=state["feature_dim"],
        seq_len=state["seq_len"],
        latent_dim=state["latent_dim"],
        hidden_dim=state["hidden_dim"],
        num_classes=state["num_classes"],
        num_layers=state["num_layers"],
        device=device,
    )
    model.load_state_dict(state["model_state_dict"])
    return model, state.get("losses")
