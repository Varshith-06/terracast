"""
GPU sequence models for RCI forecasting: LSTM, GRU, TCN, Transformer, N-BEATS
and a Temporal Fusion Transformer.

All models take an L-day window of past channels (plus, for TFT, the known-future
calendar covariates) and emit the full h = 1..H horizon in one shot (direct
multi-horizon), so no error compounding from recursive rollout.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================== windowing utilities
def make_windows(seq: np.ndarray, fut: np.ndarray, L: int, H: int,
                 origins: np.ndarray):
    """
    origins = row indices t used as forecast origins.
    Xp = the L observed days ending at t; Xf = the H known-future covariate rows.
    """
    Xp = np.stack([seq[t - L + 1: t + 1] for t in origins]).astype(np.float32)
    Xf = np.stack([fut[t + 1: t + H + 1] for t in origins]).astype(np.float32)
    return Xp, Xf


# ============================================================================ models
class LSTMForecaster(nn.Module):
    def __init__(self, n_in, H, hidden=128, layers=2, dropout=0.2, cell="lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                  nn.ReLU(), nn.Linear(hidden, H))

    def forward(self, xp, xf=None):
        out, _ = self.rnn(xp)
        return self.head(out[:, -1])


class Chomp(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return x[:, :, :-self.n] if self.n > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dil, dropout):
        super().__init__()
        pad = (k - 1) * dil
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Conv1d(cin, cout, k, padding=pad, dilation=dil)),
            Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(nn.Conv1d(cout, cout, k, padding=pad, dilation=dil)),
            Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return F.relu(self.net(x) + res)


class TCNForecaster(nn.Module):
    """Dilated causal convolutions -- receptive field covers the whole window."""
    def __init__(self, n_in, H, ch=64, levels=5, k=3, dropout=0.15):
        super().__init__()
        layers, cin = [], n_in
        for i in range(levels):
            layers.append(TCNBlock(cin, ch, k, 2 ** i, dropout))
            cin = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch, ch), nn.ReLU(),
                                  nn.Linear(ch, H))

    def forward(self, xp, xf=None):
        return self.head(self.tcn(xp.transpose(1, 2))[:, :, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_in, H, d=128, nhead=8, layers=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pos = PositionalEncoding(d)
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, H))

    def forward(self, xp, xf=None):
        z = self.enc(self.pos(self.proj(xp)))
        return self.head(z.mean(1) + z[:, -1])


class NBeatsBlock(nn.Module):
    def __init__(self, L, H, width, depth):
        super().__init__()
        mlp, cin = [], L
        for _ in range(depth):
            mlp += [nn.Linear(cin, width), nn.ReLU()]
            cin = width
        self.mlp = nn.Sequential(*mlp)
        self.back = nn.Linear(width, L)
        self.fore = nn.Linear(width, H)

    def forward(self, x):
        h = self.mlp(x)
        return self.back(h), self.fore(h)


class NBeatsForecaster(nn.Module):
    """
    Generic-architecture N-BEATS on the univariate target channel, with the
    exogenous window summarised and injected as a context vector (N-BEATSx style).
    """
    def __init__(self, n_in, H, L, target_idx=0, width=256, depth=3, blocks=4,
                 ctx=64, dropout=0.1):
        super().__init__()
        self.L, self.target_idx = L, target_idx
        self.blocks = nn.ModuleList([NBeatsBlock(L, H, width, depth) for _ in range(blocks)])
        self.ctx = nn.Sequential(nn.Flatten(), nn.Linear(L * n_in, ctx), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(ctx, H))

    def forward(self, xp, xf=None):
        x = xp[:, :, self.target_idx]
        fcast = 0.0
        for b in self.blocks:
            back, fore = b(x)
            x = x - back
            fcast = fcast + fore
        return fcast + self.ctx(xp)


class BiGRUAttn(nn.Module):
    """
    Bidirectional GRU encoder over the observed window + one attention query per
    forecast horizon.

    Bidirectionality is legitimate here because the encoder only ever sees
    seq[t-L+1 : t+1] -- every step of the window is observed at or before the
    forecast origin, so the backward pass reads no future information. The known
    future covariates are injected into the per-horizon queries rather than the
    recurrent path, which keeps that argument clean and lets each horizon know
    which calendar slot it is predicting.
    """

    def __init__(self, n_in, n_fut, H, hidden=96, layers=2, dropout=0.2, heads=4,
                 quantiles=None):
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0,
                          bidirectional=True)
        d = 2 * hidden
        self.q = nn.Parameter(torch.randn(H, d) * 0.02)
        self.fut = nn.Sequential(nn.Linear(n_fut, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        n_out = len(quantiles) if quantiles else 1
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d, n_out))
        self.H, self.quantiles = H, quantiles
        self.last_attn = None

    def forward(self, xp, xf=None):
        enc, _ = self.rnn(xp)                                  # (B, L, 2*hidden)
        q = self.q.unsqueeze(0).expand(xp.shape[0], -1, -1)
        if xf is not None:
            q = q + self.fut(xf)                               # (B, H, d)
        ctx, w = self.attn(q, enc, enc, need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()                            # (B, H, L)
        o = self.head(self.norm(ctx + q))                      # (B, H, n_out)
        return o if self.quantiles else o.squeeze(-1)


# ------------------------------------------------------------------------------ TFT
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__(); self.a = nn.Linear(d, d); self.b = nn.Linear(d, d)
    def forward(self, x): return torch.sigmoid(self.a(x)) * self.b(x)


class GateAddNorm(nn.Module):
    def __init__(self, d, dropout):
        super().__init__(); self.drop = nn.Dropout(dropout); self.glu = GLU(d); self.norm = nn.LayerNorm(d)
    def forward(self, x, skip): return self.norm(self.glu(self.drop(x)) + skip)


class GRN(nn.Module):
    """Gated Residual Network -- the TFT's universal building block."""
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.1, d_context=None):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.gan = GateAddNorm(d_out, dropout)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.fc2(F.elu(h))
        return self.gan(self.out(h), self.skip(x))


class VariableSelection(nn.Module):
    """Per-timestep softmax over input variables, each first passed through its own GRN."""
    def __init__(self, n_vars, d, dropout, d_context=None):
        super().__init__()
        self.n_vars = n_vars
        self.flat = GRN(n_vars * d, d, n_vars, dropout, d_context)
        self.var_grns = nn.ModuleList([GRN(d, d, d, dropout) for _ in range(n_vars)])

    def forward(self, x, c=None):
        # x: (B, T, n_vars, d)
        B, T, V, D = x.shape
        w = torch.softmax(self.flat(x.reshape(B, T, V * D), c), dim=-1)
        proc = torch.stack([g(x[:, :, i]) for i, g in enumerate(self.var_grns)], dim=2)
        return (proc * w.unsqueeze(-1)).sum(2), w


class InterpretableMHA(nn.Module):
    """Multi-head attention with heads sharing one value projection (TFT eq. 14-16)."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, self.dk)
        self.out = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        qs = self.q(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        ks = self.k(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        vs = self.v(v).unsqueeze(1)
        att = (qs @ ks.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            att = att.masked_fill(mask, float("-inf"))
        att = self.drop(torch.softmax(att, -1))
        head_out = att @ vs
        return self.out(self.drop(head_out.mean(1))), att


class TFT(nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., 2021), single-series configuration.

    Encoder consumes the L past observed channels; decoder consumes the H known
    future calendar covariates. Static context is a learned site embedding.
    Trained with quantile loss; the 0.5 quantile is the point forecast.
    """
    def __init__(self, n_past, n_future, H, d=64, heads=4, dropout=0.15,
                 quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.H, self.quantiles = H, list(quantiles)
        self.n_past, self.n_future = n_past, n_future

        self.static = nn.Parameter(torch.zeros(1, d))
        self.ctx_sel = GRN(d, d, d, dropout)
        self.ctx_enr = GRN(d, d, d, dropout)
        self.ctx_h = GRN(d, d, d, dropout)
        self.ctx_c = GRN(d, d, d, dropout)

        self.past_emb = nn.ModuleList([nn.Linear(1, d) for _ in range(n_past)])
        self.fut_emb = nn.ModuleList([nn.Linear(1, d) for _ in range(n_future)])
        self.vs_past = VariableSelection(n_past, d, dropout, d)
        self.vs_fut = VariableSelection(n_future, d, dropout, d)

        self.enc_lstm = nn.LSTM(d, d, batch_first=True)
        self.dec_lstm = nn.LSTM(d, d, batch_first=True)
        self.gan_lstm = GateAddNorm(d, dropout)
        self.enrich = GRN(d, d, d, dropout, d)
        self.attn = InterpretableMHA(d, heads, dropout)
        self.gan_attn = GateAddNorm(d, dropout)
        self.ff = GRN(d, d, d, dropout)
        self.gan_ff = GateAddNorm(d, dropout)
        self.out = nn.Linear(d, len(self.quantiles))

    def forward(self, xp, xf):
        B = xp.shape[0]
        s = self.static.expand(B, -1)
        c_sel, c_enr = self.ctx_sel(s), self.ctx_enr(s)
        c_h, c_c = self.ctx_h(s), self.ctx_c(s)

        ep = torch.stack([e(xp[:, :, i: i + 1]) for i, e in enumerate(self.past_emb)], 2)
        ef = torch.stack([e(xf[:, :, i: i + 1]) for i, e in enumerate(self.fut_emb)], 2)
        zp, _ = self.vs_past(ep, c_sel.unsqueeze(1))
        zf, w_fut = self.vs_fut(ef, c_sel.unsqueeze(1))

        h0 = c_h.unsqueeze(0).contiguous(); c0 = c_c.unsqueeze(0).contiguous()
        enc, (hn, cn) = self.enc_lstm(zp, (h0, c0))
        dec, _ = self.dec_lstm(zf, (hn, cn))
        seq = torch.cat([enc, dec], 1)
        seq = self.gan_lstm(seq, torch.cat([zp, zf], 1))
        enr = self.enrich(seq, c_enr.unsqueeze(1))

        T = seq.shape[1]
        L = xp.shape[1]
        # decoder step i may attend to everything up to and including itself
        idx = torch.arange(T, device=xp.device)
        mask = (idx.unsqueeze(0) > idx.unsqueeze(1))[L:]        # (H, T)
        att_out, _ = self.attn(enr[:, L:], enr, enr, mask.unsqueeze(0).unsqueeze(0))
        x = self.gan_attn(att_out, enr[:, L:])
        x = self.gan_ff(self.ff(x), seq[:, L:])
        return self.out(x)                                       # (B, H, n_quantiles)


def quantile_loss(pred, target, quantiles):
    # pred (B,H,Q), target (B,H)
    t = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    e = t - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


# ========================================================================== training
def train_model(model, data, epochs=300, lr=1e-3, batch=128, patience=30,
                weight_decay=1e-4, quantiles=None, verbose=False, seed=0):
    """data = dict of (Xp, Xf, Y) tensors already on device for train/val/test."""
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, math.ceil(len(data["train"][0]) / batch)),
        pct_start=0.2)
    Xp, Xf, Y = data["train"]
    vXp, vXf, vY = data["val"]
    n = len(Xp)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator(device="cpu"); g.manual_seed(seed)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, batch):
            idx = perm[i: i + batch]
            opt.zero_grad(set_to_none=True)
            out = model(Xp[idx], Xf[idx])
            loss = (quantile_loss(out, Y[idx], quantiles) if quantiles
                    else F.mse_loss(out, Y[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            out = model(vXp, vXf)
            vloss = (quantile_loss(out, vY, quantiles) if quantiles
                     else F.mse_loss(out, vY)).item()
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"    ep {ep:3d}  val {vloss:.5f}  best {best:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, ep + 1


@torch.no_grad()
def predict(model, Xp, Xf, quantiles=None, batch=512):
    model.eval()
    outs = []
    for i in range(0, len(Xp), batch):
        o = model(Xp[i: i + batch], Xf[i: i + batch])
        if quantiles:
            o = o[:, :, quantiles.index(0.5)]
        outs.append(o.float().cpu())
    return torch.cat(outs).numpy()
