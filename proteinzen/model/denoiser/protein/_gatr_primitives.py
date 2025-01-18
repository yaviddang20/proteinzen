import torch

from gatr.primitives.bilinear import _load_bilinear_basis
from gatr.primitives.dual import _compute_efficient_join
from gatr.utils.einsum import gatr_einsum


def weighted_geometric_product(W: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product f(x,y) = xy.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    gp = _load_bilinear_basis("gp", x.device, x.dtype)
    weighted_gp = W * gp[None]

    # Compute geometric product
    outputs = gatr_einsum("n i j k, ... n j, ... n k -> ... n i", weighted_gp, x, y)

    return outputs


def weighted_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer product `f(x,y) = x ^ y`.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    op = _load_bilinear_basis("outer", x.device, x.dtype)

    # Compute geometric product
    outputs = gatr_einsum("i j k, ... j, ... k -> ... i", op, x, y)

    return outputs

def weighted_equivariant_join(
    W: torch.Tensor, x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Computes the equivariant join, using the efficient implementation.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Rquivariant join result.
    """

    kernel = _compute_efficient_join(x.device, x.dtype)
    weighted_kernel = W * kernel[None]
    return reference[..., [14]] * gatr_einsum("n i j k , ... n j, ... n k -> ... n i", weighted_kernel, x, y)
