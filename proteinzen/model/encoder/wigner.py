""" Wigner D matrix calculations

Taken from e3nn, modified so all operations can run on gpu

"""
import math
from typing import Tuple, Optional

import torch
from e3nn.o3 import Irrep, Irreps
from e3nn.o3 import _rotation


def _torch_get_default_dtype() -> torch.dtype:
    """A torchscript-compatible version of torch.get_default_dtype()"""
    return torch.empty(0).dtype


def torch_get_default_device() -> torch.device:
    return torch.empty(0).device

def explicit_default_types(dtype: Optional[torch.dtype], device: Optional[torch.device]) -> Tuple[torch.dtype, torch.device]:
    """A torchscript-compatible type resolver"""
    if dtype is None:
        dtype = _torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device

def su2_generators(j) -> torch.Tensor:
    m = torch.arange(-j, j)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    m = torch.arange(-j + 1, j + 1)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    m = torch.arange(-j, j + 1)
    return torch.stack(
        [
            0.5 * (raising + lowering),  # x (usually)
            torch.diag(1j * m),  # z (usually)
            -0.5j * (raising - lowering),  # -y (usually)
        ],
        dim=0,
    )

def change_basis_real_to_complex(l: int, dtype=None, device=None) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2**0.5
    q = (-1j) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    dtype, device = explicit_default_types(dtype, device)
    dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    # make sure we always get:
    # 1. a copy so mutation doesn't ruin the stored tensors
    # 2. a contiguous tensor, regardless of what transpositions happened above
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)

def so3_generators(l) -> torch.Tensor:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = torch.conj(Q.T) @ X @ Q
    assert torch.all(torch.abs(torch.imag(X)) < 1e-5)
    return torch.real(X)

def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = so3_generators(l).to(alpha.device)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])


def irrep_D_from_angles(irrep: Irrep, alpha, beta, gamma, k=None):
    r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

    (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`
        How many times the parity is applied.

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 2l+1, 2l+1)`

    See Also
    --------
    o3.wigner_D
    Irreps.D_from_angles
    """
    if k is None:
        k = torch.zeros_like(alpha)

    alpha, beta, gamma, k = torch.broadcast_tensors(alpha, beta, gamma, k)
    return wigner_D(irrep.l, alpha, beta, gamma) * irrep.p ** k[..., None, None]

def irrep_D_from_quaternion(irrep: Irrep, q, k=None):
    r"""Matrix of the representation

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
    """
    return irrep_D_from_angles(irrep, *_rotation.quaternion_to_angles(q), k)


def direct_sum(*matrices):
    r"""Direct sum of matrices, put them in the diagonal"""
    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i : i + m, j : j + n] = x
        i += m
        j += n
    return out

def irreps_D_from_angles(irreps: Irreps, alpha, beta, gamma, k=None):
    r"""Matrix of the representation

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
    """
    return direct_sum(*[irrep_D_from_angles(ir, alpha, beta, gamma, k) for mul, ir in irreps for _ in range(mul)])

def irreps_D_from_quaternion(irreps: Irreps, q, k=None):
    r"""Matrix of the representation

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
    """
    return irreps_D_from_angles(irreps, *_rotation.quaternion_to_angles(q), k)

def fast_wigner_D_rotation(irreps, quat, features):
    out = []
    for (m, ir), sl in zip(irreps, irreps.slices()):
        if ir.l == 0:
            out.append(features[..., sl])
        else:
            wigner_D = irrep_D_from_quaternion(ir, quat)
            irrep_features = features[..., sl].view(*features.shape[:-1], m, -1)
            irrep_features = torch.einsum("...ij,...kj->...ki", wigner_D, irrep_features)
            out.append(irrep_features.reshape(*features.shape[:-1], -1))
    return torch.cat(out, dim=-1)