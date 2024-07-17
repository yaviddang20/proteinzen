import torch
import geomstats.algebra_utils as utils

def to_tangent(metric, vector, base_point, eps=1e-6):
    """Project a vector to the tangent space.

    Project a vector in Euclidean space
    on the tangent space of the hypersphere at a base point.

    Parameters
    ----------
    vector : array-like, shape=[..., dim + 1]
        Vector in Euclidean space.
    base_point : array-like, shape=[..., dim + 1]
        Point on the hypersphere defining the tangent space,
        where the vector will be projected.

    Returns
    -------
    tangent_vec : array-like, shape=[..., dim + 1]
        Tangent vector in the tangent space of the hypersphere
        at the base point.
    """
    sq_norm = torch.sum(base_point**2, axis=-1)
    inner_prod = metric.inner_product(base_point, vector)
    coef = inner_prod / (sq_norm + eps)
    return vector - torch.einsum("...,...j->...j", coef, base_point)


def exp(metric, tangent_vec, base_point):
    """Compute the Riemannian exponential of a tangent vector.

    Parameters
    ----------
    tangent_vec : array-like, shape=[..., dim + 1]
        Tangent vector at a base point.
    base_point : array-like, shape=[..., dim + 1]
        Point on the hypersphere.

    Returns
    -------
    exp : array-like, shape=[..., dim + 1]
        Point on the hypersphere equal to the Riemannian exponential
        of tangent_vec at the base point.
    """
    proj_tangent_vec = to_tangent(metric, tangent_vec, base_point)
    norm2 = metric.squared_norm(proj_tangent_vec)

    coef_1 = utils.taylor_exp_even_func(norm2, utils.cos_close_0, order=4)
    coef_2 = utils.taylor_exp_even_func(norm2, utils.sinc_close_0, order=4)
    _exp = torch.einsum("...,...j->...j", coef_1, base_point) + torch.einsum(
        "...,...j->...j", coef_2, proj_tangent_vec
    )

    return _exp


def log(metric, point, base_point, eps=1e-6):
    """Compute the Riemannian logarithm of a point.

    Parameters
    ----------
    point : array-like, shape=[..., dim + 1]
        Point on the hypersphere.
    base_point : array-like, shape=[..., dim + 1]
        Point on the hypersphere.

    Returns
    -------
    log : array-like, shape=[..., dim + 1]
        Tangent vector at the base point equal to the Riemannian logarithm
        of point at the base point.
    """
    inner_prod = metric.inner_product(base_point, point)
    cos_angle = torch.clip(inner_prod, -1.0 + eps, 1.0 - eps)
    squared_angle = torch.arccos(cos_angle) ** 2
    coef_1_ = utils.taylor_exp_even_func(
        squared_angle, utils.inv_sinc_close_0, order=5
    )
    coef_2_ = utils.taylor_exp_even_func(
        squared_angle, utils.inv_tanc_close_0, order=5
    )
    _log = torch.einsum("...,...j->...j", coef_1_, point) - torch.einsum(
        "...,...j->...j", coef_2_, base_point
    )
    return _log