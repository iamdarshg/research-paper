"""Small, checkpoint-friendly mHC-style residual routing primitive.

This is the production version of the plain arm from the 16^3 probe: an
identity-initialized learned stream mixer projected to the positive doubly
stochastic manifold with Sinkhorn iterations.  It deliberately contains no
RMS normalization or gate, because those variants were the weaker arms in the
local comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _stream_count(channels: int, requested: int) -> int:
    requested = max(1, min(int(requested), int(channels)))
    if channels % requested == 0:
        return requested
    divisors = [candidate for candidate in range(requested, 0, -1) if channels % candidate == 0]
    return divisors[0] if divisors else 1


def sinkhorn_doubly_stochastic(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """Project routing logits to a positive approximately doubly-stochastic map."""
    matrix = logits.float().clamp(-20.0, 20.0).exp()
    for _ in range(max(1, int(iterations))):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)
    return matrix.to(dtype=logits.dtype)


class ManifoldHyperConnection(nn.Module):
    """Identity-initialized constrained residual stream mixer.

    The module accepts channel-first convolution tensors (`N,C,*`) and
    channel-last linear tensors (`N,C`).  It routes the residual/update branch
    only; callers retain the ordinary residual skip path.  If a channel count
    is not divisible by the requested stream count, the largest compatible
    divisor is selected and recorded in the module metadata.
    """

    def __init__(
        self,
        channels: int,
        *,
        streams: int = 8,
        sinkhorn_iterations: int = 8,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.streams = _stream_count(self.channels, streams)
        self.channels_per_stream = self.channels // self.streams
        self.sinkhorn_iterations = max(1, int(sinkhorn_iterations))
        self.enabled = bool(enabled)
        init = torch.full((self.streams, self.streams), -4.0)
        diagonal = torch.arange(self.streams)
        init[diagonal, diagonal] = 4.0
        self.routing_logits = nn.Parameter(init)

    def routing(self) -> torch.Tensor:
        return sinkhorn_doubly_stochastic(self.routing_logits, self.sinkhorn_iterations)

    def forward(self, update: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.streams == 1:
            return update
        if update.ndim < 2:
            raise ValueError("mHC updates must have at least two dimensions")
        routing = self.routing()
        if update.ndim == 2:
            shape = update.shape
            streams = update.reshape(-1, self.streams, self.channels_per_stream)
            mixed = torch.einsum("ij,njc->nic", routing, streams)
            return mixed.reshape(shape)

        # Convolutional tensors are channel-first.  Move channels to the last
        # axis so the same stream operation is used for every spatial layout.
        channel_last = update.movedim(1, -1)
        shape = channel_last.shape
        streams = channel_last.reshape(-1, self.streams, self.channels_per_stream)
        mixed = torch.einsum("ij,njc->nic", routing, streams)
        return mixed.reshape(shape).movedim(-1, 1)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, streams={self.streams}, "
            f"sinkhorn_iterations={self.sinkhorn_iterations}, enabled={self.enabled}"
        )


def load_state_dict_mhc_compatible(
    module: nn.Module,
    state_dict: dict,
    *,
    allow_missing_prefixes: tuple[str, ...] = (),
):
    """Load a checkpoint while allowing an older checkpoint to omit mHC keys.

    mHC is opt-in, and its routing logits are identity-initialized.  That makes
    it safe to warm-start an enabled module from a pre-mHC checkpoint, but it
    is not safe to silently discard learned mHC parameters when loading into a
    disabled module.  All non-mHC incompatibilities remain hard errors.
    """
    result = module.load_state_dict(state_dict, strict=False)
    missing_non_mhc = [
        key
        for key in result.missing_keys
        if "mhc" not in key.lower()
        and not any(key.startswith(prefix) for prefix in allow_missing_prefixes)
    ]
    unexpected_mhc = [key for key in result.unexpected_keys if "mhc" in key.lower()]
    unexpected_non_mhc = [key for key in result.unexpected_keys if "mhc" not in key.lower()]
    if missing_non_mhc or unexpected_mhc or unexpected_non_mhc:
        details = []
        if missing_non_mhc:
            details.append(f"missing={missing_non_mhc}")
        if unexpected_mhc:
            details.append(
                "checkpoint contains learned mHC parameters but the current "
                f"module cannot accept them: {unexpected_mhc}"
            )
        if unexpected_non_mhc:
            details.append(f"unexpected={unexpected_non_mhc}")
        raise RuntimeError("Incompatible checkpoint state: " + "; ".join(details))
    return result
