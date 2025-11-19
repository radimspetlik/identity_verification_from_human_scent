import math
from pathlib import Path
from typing import Union

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

# ----------------------------------------------------------------------------- #
#                          GAUSSIAN KERNELS                         #
# ----------------------------------------------------------------------------- #

class GaussModel(torch.nn.Module):
    def __init__(self, centers, **_):
        super(GaussModel, self).__init__()
        self.K = len(centers)
        self.centers = centers.clone().detach().to(dtype=torch.float32).view(self.K, 1)
        self.sigmas = torch.nn.Parameter(torch.ones(self.K, 1))
        self.multipliers = torch.nn.Parameter(torch.ones(self.K, 1))

    def forward(self, positions):
        """
        positions: Tensor of shape (num_samples, N) or (N,) where N is the number of mass values
        intensities: Tensor of shape (num_samples, N) or (N,)

        Returns:
            Tensor of shape (K, num_samples, N) or (K, N), each row corresponds to kernel weights of each K
        """
        if positions.dim() == 1:
            positions = positions.view(1, -1)  # (1, N)

        # Ensure shapes are broadcast-friendly
        # positions: (num_samples, N) -> (1, num_samples, N)
        positions = positions.unsqueeze(0)  # Shape: (1, num_samples, N)

        # centers, sigmas, multipliers: (K, 1, 1)
        centers = self.centers.view(-1, 1, 1)  # (K, 1, 1)
        sigmas = self.sigmas.view(-1, 1, 1)  # (K, 1, 1)
        multipliers = self.multipliers.view(-1, 1, 1)  # (K, 1, 1)

        # Compute Gaussian kernel in batch
        T = (1 / (sigmas * torch.sqrt(torch.tensor(2 * np.pi, device=sigmas.device)))) \
            * torch.exp(-0.5 * ((positions - centers) / sigmas) ** 2) * multipliers

        return T

    def get_sigmas_multipliers(self):
        return self.sigmas, self.multipliers

    @torch.no_grad()
    def plot(self, path: Union[Path, str], points: int = 1000) -> None:
        """
        Save a figure with all learned Gaussians (amplitude‑scaled).
        Supports both GaussianBank and GaussModel.
        """

        centers = self.centers.squeeze().cpu()
        sigmas = self.sigmas.squeeze().cpu()
        amps = self.multipliers.squeeze().cpu()

        plt.figure(figsize=(12, 3))
        for mu_t, sigma_t, a_t in zip(centers, sigmas, amps):
            x_t = torch.linspace(mu_t - 3 * sigma_t, mu_t + 3 * sigma_t, points)
            y_t = (a_t / (sigma_t * math.sqrt(2 * math.pi))) * torch.exp(
                -0.5 * ((x_t - mu_t) / sigma_t) ** 2
            )
            plt.plot(x_t.cpu().numpy(), y_t.cpu().numpy())

        plt.tight_layout()
        plt.title("Learned Gaussian Kernels")
        plt.xlabel("m/z")
        plt.ylabel("Amplitude")
        plt.xlim(0, 800)
        plt.grid(True)
        plt.savefig(path)
        plt.close()


class GaussianBank(nn.Module):
    def __init__(self, centers: torch.Tensor, eps: float = 1e-2, **_):
        """
        centers : float32 [K] – fixed μ_k.
        eps     : lower bound that σ will never cross.
        """
        super().__init__()
        # keep original order for forward()
        self.register_buffer("centers", centers.view(-1, 1, 1))  # [K,1,1]

        # --- smart σ initialisation -----------------------------------------
        with torch.no_grad():
            # sort to compute neighbour spacings
            sorted_mu, sort_idx = torch.sort(centers.flatten())
            delta = torch.empty_like(sorted_mu)
            delta[0]         = sorted_mu[1]   - sorted_mu[0]
            delta[1:-1]      = torch.min(sorted_mu[1:-1] - sorted_mu[:-2], sorted_mu[2:]   - sorted_mu[1:-1])
            delta[-1]        = sorted_mu[-1]  - sorted_mu[-2]
            sigma0       = (delta / (8.0 * torch.log(torch.tensor(2.0))).sqrt()).clamp_min(eps)                   # [K]
            # undo the sort so kernels keep user‑supplied order
            unsorted_sigma0 = sigma0[sort_idx.argsort()].view(-1, 1, 1)

        self.raw_sigma = nn.Parameter(inv_softplus(unsorted_sigma0 - eps))
        self.raw_amp   = nn.Parameter(torch.zeros_like(self.raw_sigma))
        self.eps = eps

    # --------------------------------------------------------------------- #
    @property
    def sigma(self):       # strictly positive
        return self.eps + torch.nn.functional.softplus(self.raw_sigma)

    @property
    def amp(self):
        return torch.nn.functional.softplus(self.raw_amp)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:  # [B,N] -> [K,B,N]
        pos = positions.unsqueeze(0)                           # [1,B,N]
        sigma   = self.sigma                                       # [K,1,1]
        coef = 1. / (sigma * (2*np.pi)**0.5)
        exponent = -0.5 * ((pos - self.centers) / sigma) ** 2
        return self.amp * coef * torch.exp(exponent)

    @torch.no_grad()
    def plot(self, path: Union[Path, str], points: int = 1000) -> None:
        """
        Save a figure with all learned Gaussians (amplitude‑scaled).
        Supports both GaussianBank and GaussModel.
        """
        centers = self.centers.squeeze().cpu()
        sigmas = self.sigma.squeeze().cpu()
        amps = self.amp.squeeze().cpu()

        plt.figure(figsize=(12, 3))
        for mu_t, sigma_t, a_t in zip(centers, sigmas, amps):
            x_t = torch.linspace(mu_t - 3 * sigma_t, mu_t + 3 * sigma_t, points)
            y_t = (a_t / (sigma_t * math.sqrt(2 * math.pi))) * torch.exp(
                -0.5 * ((x_t - mu_t) / sigma_t) ** 2
            )
            plt.plot(x_t.cpu().numpy(), y_t.cpu().numpy())

        plt.tight_layout()
        plt.title("Learned Gaussian Kernels")
        plt.xlabel("m/z")
        plt.ylabel("Amplitude")
        plt.xlim(0, 800)
        plt.grid(True)
        plt.savefig(path)
        plt.close()


# --------------------------------------------------------------------------- #
#  TriangleBank
# --------------------------------------------------------------------------- #
class TriangleBank(nn.Module):
    r"""Bank of piece‑wise linear **triangular kernels**

        k(x; μ_k, w_k) = a_k · max(1 − |x − μ_k| / w_k, 0)

    Parameters
    ----------
    centers : torch.Tensor, shape [K]
        Fixed triangle apices μ_k.
    eps : float, default 1e‑3
        Positive lower bound the half‑width ``w_k`` will never cross.

    Shapes
    ------
    forward(positions[B, N]) → [K, B, N]
    """

    def __init__(self, centers: torch.Tensor, eps: float = 1e-3, **_):
        super().__init__()

        # ---------- fixed centres ------------------------------------------------
        self.centers = centers.to(torch.float32).view(-1, 1, 1)

        # ---------- smart half‑width initialisation -----------------------------
        # Use neighbour spacing so that each triangle vanishes at the next centre
        with torch.no_grad():
            sorted_mu, sort_idx = torch.sort(centers.flatten())      # [K]
            delta = torch.empty_like(sorted_mu)                      # [K]
            delta[0]      = sorted_mu[1]   - sorted_mu[0]
            delta[1:-1]   = torch.min(sorted_mu[1:-1] - sorted_mu[:-2],
                                       sorted_mu[2:]   - sorted_mu[1:-1])
            delta[-1]     = sorted_mu[-1]  - sorted_mu[-2]

            w0 = delta.clamp_min(eps)                                # half‑width
            w0 = w0[sort_idx.argsort()].view(-1, 1, 1)               # unsort

        # ---------- learnable parameters (soft‑plus for positivity) -------------
        self.raw_w   = nn.Parameter(inv_softplus(w0 - eps))          # half‑widths
        self.raw_amp = nn.Parameter(inv_softplus(torch.ones_like(w0)))  # start a_k≈1
        self.eps     = eps

    # ------------------------------ properties ----------------------------------
    @property
    def w(self) -> torch.Tensor:          # strictly positive half‑widths [K,1,1]
        return self.eps + F.softplus(self.raw_w)

    @property
    def amp(self) -> torch.Tensor:        # strictly positive amplitudes [K,1,1]
        return F.softplus(self.raw_amp)

    # -------------------------------- forward -----------------------------------
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        positions : torch.Tensor, shape [B, N]

        Returns
        -------
        torch.Tensor, shape [K, B, N] – triangular kernel responses
        """
        # Ensure positions are in the right shape (B, N) or (N,)
        if positions.dim() == 1:
            positions = positions.view(1, -1)  # (1, N)

        pos   = positions.unsqueeze(0)                        # [1,B,N]
        diff  = (pos - self.centers).abs()                    # [K,B,N]
        core  = torch.relu(1.0 - diff / self.w)               # triangular shape
        response = self.amp * core                             # [K,B,N]
        return response

    @torch.no_grad()
    def plot(self,
             path: Union[Path, str],
             points: int = 1000,
             xlim: tuple[float, float] | None = (0, 800)) -> None:
        """
        Save a figure with all learned triangular kernels (amplitude‑scaled).

        Parameters
        ----------
        path    : str | Path
            Destination for the PNG/PDF/… image.
        points  : int, default 1000
            Samples per kernel curve.
        xlim    : tuple(float,float) | None
            Optional fixed x‑axis span.  Pass *None* to auto‑scale.
        """
        centers = self.centers.squeeze().cpu()    # [K]
        widths  = self.w.squeeze().cpu()          # half‑widths
        amps    = self.amp.squeeze().cpu()        # amplitudes

        fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Top: linear plot
        for mu, w, a in zip(centers, widths, amps):
            x_t = torch.linspace(mu - w, mu + w, points)
            y_t = a * torch.relu(1.0 - (x_t - mu).abs() / w)
            ax_lin.plot(x_t.numpy(), y_t.numpy(), label=f"μ={mu:.2f}")

        ax_lin.set_ylabel("Amplitude")
        ax_lin.set_title("Learned Triangular Kernels (linear scale)")
        ax_lin.grid(True)

        # Bottom: semilogy
        for mu, w, a in zip(centers, widths, amps):
            x_t = torch.linspace(mu - w, mu + w, points)
            y_t = a * torch.relu(1.0 - (x_t - mu).abs() / w)
            ax_log.semilogy(x_t.numpy(), y_t.numpy())

        ax_log.set_xlabel("m/z")
        ax_log.set_ylabel("Amplitude (log scale)")
        ax_log.set_title("Learned Triangular Kernels (log scale)")
        ax_log.grid(True)

        # X‐limits if specified
        if xlim is not None:
            ax_lin.set_xlim(*xlim)
            ax_log.set_xlim(*xlim)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


class SumBank(nn.Module):
    r"""Single-channel 'kernel' that reduces to a sum over values.

    Rationale
    ---------
    If your downstream aggregates with something like:
        feats = torch.einsum('kbn,bn->kb', response, values)
    then setting response[k,b,n] = 1 for all n makes feats[k,b] = sum_n values[b,n].

    Shapes
    ------
    forward(positions[B, N]) -> [K, B, N]  (all ones)
        By default K = 1. If you want to match an existing K (e.g. len(centers)),
        pass centers and set `replicate_to_centers=True`.
    """

    def __init__(self, *_, **__):
        super().__init__()
        self.K = 1
        self.centers = nn.Parameter(torch.tensor(self.K).to(torch.float32).view(-1, 1, 1))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # Accept [N] or [B,N]
        if positions.dim() == 1:
            positions = positions.view(1, -1)  # (1, N)
        B, N = positions.shape
        # Create [1,B,N] ones on the same device/dtype, then (optionally) replicate to K
        ones = positions.new_ones((1, B, N))
        return ones.expand(self.K, -1, -1)
