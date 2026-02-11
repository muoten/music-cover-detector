import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-12, precision: str = "high"):
    """L2 normalize the input tensor. In the case that the norm is small,
    a small value is added to the norm to avoid division by zero."""

    assert x.dim() == 2, "Input tensor must be 2D"
    assert precision in ["high", "mid", "low"], "Invalid precision value"

    norms = torch.norm(x, p=2, dim=1, keepdim=True)

    if precision == "high":
        norms = norms + ((norms == 0).type_as(norms) * eps)
    elif precision == "mid":
        norms = torch.clamp(norms, min=eps)
    elif precision == "low":
        norms = norms + eps

    x = x / norms

    return x
