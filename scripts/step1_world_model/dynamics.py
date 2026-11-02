from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn


class TimeFourierEncoding(nn.Module):
    """Simple Fourier features for scalar time input.

    Parameters
    ----------
    n_freqs : int
        Number of frequency components.
    scale : float
        Global frequency scale.
    """

    def __init__(self, n_freqs: int = 4, scale: float = 1.0) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        freqs = torch.linspace(1.0, float(n_freqs), n_freqs) * float(scale)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return self.n_freqs * 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode scalar time *t* into Fourier features.

        Parameters
        ----------
        t : ``(B,)`` or scalar tensor.

        Returns
        -------
        features : ``(B, 2 * n_freqs)`` tensor.
        """
        if t.dim() == 0:
            t = t.view(1)
        t = t.view(-1, 1) * self.freqs.view(1, -1)  # (B, n_freqs)
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class DriftNet(nn.Module):
    """Drift network f(x, t[, cond]) for dx/dt.

    向后兼容: ``cond_dim`` 默认 0, 此时 ``forward(x, t)`` 行为与原版完全一致。
    条件版: 设 ``cond_dim>0``, ``forward(x, t, cond)`` 将 cond 拼到输入,
    用于跨物种/跨组织的条件向量场 (例如四象限世界模型)。
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        depth: int = 3,
        time_freq: int = 4,
        dropout: float = 0.0,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.time_enc = TimeFourierEncoding(time_freq)
        in_dim = dim + self.time_enc.out_dim + cond_dim

        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])
        t_enc = self.time_enc(t)  # (B, 2 * n_freqs)
        parts = [x, t_enc]
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(
                    "DriftNet was built with cond_dim>0; forward requires cond"
                )
            if cond.dim() == 1:
                cond = cond.unsqueeze(0).expand(x.shape[0], -1)
            parts.append(cond)
        h = torch.cat(parts, dim=-1)
        h = self.mlp(h)
        return self.out(h)


# =========================
# ODE integrator
# =========================

def ode_step_euler(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One explicit Euler step."""
    dx = f(x, t)
    x_next = x + dt * dx
    t_next = t + dt
    return x_next, t_next


def ode_step_rk4(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One classical Runge–Kutta (RK4) step."""
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    t_next = t + dt
    return x_next, t_next


def integrate_ode(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    n_steps: int,
    method: str = "rk4",
) -> torch.Tensor:
    """Integrate x' = f(x, t) from t0 to t1.

    注意: f 的签名是 (x, t) -> dx。若 DriftNet 需要 cond, 由调用方在 f
    的闭包里捕获 cond, 例如 ``lambda x, t: drift(x, t, cond)``。这样本
    积分器无需感知 cond, 完全向后兼容。
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    method = method.lower()
    if method == "rk4":
        step_fn = ode_step_rk4
    elif method == "euler":
        step_fn = ode_step_euler
    else:
        raise ValueError(f"Unknown ODE method: {method!r}")

    dt = (t1 - t0) / float(n_steps)
    x = x0
    t = t0
    for _ in range(n_steps):
        x, t = step_fn(f, x, t, dt)
    return x
