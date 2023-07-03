""" Helper functions for zernike transforms in pytorch

Parts adapted from https://github.com/StatPhysBio/protein_holography/blob/main/protein_holography/hologram/hologram.py """

import e3nn.o3 as o3
import numpy as np
import torch
import scipy.special




class Hyp2f1(torch.autograd.Function):
    """ A wrapper for scipy.special.hyp2f1
        Only tracks gradients through z """
    @staticmethod
    def forward(ctx, a, b, c, z):
        """ Forward pass for the ordinary hypergeometric function

        See scipy.special.hyp2f1 for usage
        """
        a_, b_, c_ = map(torch.tensor, [a, b, c])
        ctx.save_for_backward(a_, b_, c_, z)
        z = z.numpy(force=True)
        return torch.from_numpy(scipy.special.hyp2f1(a, b, c, z))

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass for the ordinary hypergeometric function

        See scipy.special.hyp2f1 for usage
        """
        a, b, c, z = ctx.saved_tensors
        prefactor = a * b / c
        a, b, c, z = map(lambda x: x.numpy(force=True), [a, b, c, z])
        forward = scipy.special.hyp2f1(a+1, b+1, c+1, z)

        return None, None, None, grad_output * prefactor * torch.from_numpy(forward)

### allow for access as a regular function
hyp2f1 = Hyp2f1.apply


def _binom(x, y):
    """ Compute the generalized binomial coefficient """
    log_fac_x = torch.lgamma(x+1)  # log n!
    log_fac_y = torch.lgamma(y+1)  # log k!
    log_fac_xmy = torch.lgamma(x-y+1)  # log (n-k)!
    return torch.exp(log_fac_x - log_fac_y - log_fac_xmy)  # binom(x, y)


# adapted from https://github.com/StatPhysBio/protein_holography/blob/main/protein_holography/hologram/hologram.py
def zernike_radial(n, l, r_scale):
    def radial(r):
        """
        Args
        ----
        r : torch.Tensor (n_batch x n_res x n_points x 3)

        Returns
        -------
        c : torch.tensor
        """
        if (n-l) % 2 == 1:
            return torch.zeros(r.shape[:-1])

        norm_r = torch.linalg.norm(r, dim=-1) / r_scale  # (n_batch x n_res x n_points)
        # dimension of the Zernike polynomial
        D = 3
        # constituent terms in the polynomial
        A = np.power(-1,(n-l)/2)
        B = np.sqrt(2*n + D)
        C = scipy.special.binom((n+l+D)/2 - 1,
                                (n-l)/2)
        E = hyp2f1(-(n-l)/2,
                   (n+l+D)/2,
                   l+D/2,
                   norm_r*norm_r)
        F = torch.pow(norm_r,l)

        # assemble coefficients
        radial_out = A * B * C * E * F

        return radial_out  # (n_batch x n_res x n_points)
    return radial


class ZernikeTransform:
    """ A cache for Zernike radial polynomials """
    def __init__(self, n_max=None, r_scale=None):
        self.cache = None
        if n_max and r_scale:
            self.init(n_max, r_scale)

    def get(self, n, l):
        if self.cache is None:
            raise Exception("ZernikeTransform has not been initialized. Use .init(n_max, r_scale)")
        return self.cache[(n, l)]

    def init(self, n_max, r_scale):
        self.cache = {}
        self.n_max = n_max
        self.r_scale = r_scale
        for n in range(n_max):
            for l in range(n+1):
                self.cache[(n, l)] = zernike_radial(n, l, r_scale)

    def forward_transform(self, points, points_mask, point_value=None):
        """ Compute the zernike polynomial expansion coefficients for a point cloud.
        The points are treated as a sum of Dirac distributions.

        Args
        -----
        points : torch.Tensor (n_batch x n_res x n_points x 3)
            point cloud to encode
        n_mask : int
            max n value to expand to. larger n = higher resolution approximation
        points_mask : torch.Tensor (n_batch x n_res x n_points)
            point cloud mask
        point_value: None, int, or torch.Tensor (n_batch x n_res x n_channel x n_points)
            scalr value per point, can be channeled
        """
        if point_value is None:
            point_value = 1
        points_mask = points_mask.unsqueeze(-2)  # add channel

        Z = {}
        for n in range(self.n_max):
            for l in range(n+1):
                # we retain the zeros since they're used in se3 transformers
                if (n-l) % 2 == 1:
                    Z[(n, l)] = torch.zeros(list(points_mask.shape[:-1]) + [2*l+1])

                R = self.get(n, l)
                y = o3.spherical_harmonics(l, points, normalize=True)  # (n_batch x n_res x n_points x 2l+1)
                r = R(points)  # (n_batch x n_res x n_points)

                # add channel
                y = y.unsqueeze(-3)
                r = r.unsqueeze(-2)
                # fix shapes for broadcasting
                r = r.unsqueeze(-1)
                c_nl = (y * r * point_value)  # n_batch x n_res x n_channel x n_points x 2l+1
                c_nl[points_mask] = 0
                if c_nl.isnan().any():
                    select = c_nl.isnan().any(dim=-1)
                    print(points[select.squeeze()])
                    print(y[select])
                    print(r[select])
                    print(points_mask[select])
                    exit()

                Z[(n, l)] = c_nl.sum(-2)

        return Z  # Dict[Tuple[int, int], Tensor[n_batch x n_res x n_channel x 2l+1]]

    def back_transform(self, Z):
        """ Reverse a zernike transform

        Args
        -----
        Z : Dict
            a dictionary of zernike polynomial expansion coefficients

        Returns
        -------
        rho : func: R^3 -> R
            density function
        """
        def rho(x):
            """ Density function
            Args
            ----
            x : torch.Tensor (n_batch x n_res x n_channel x n_points x 3)
            """
            ret = 0
            for (n, l), c_nl in Z.items():
                if (n-l) % 2 == 1:
                    continue
                R = self.get(n, l)
                y = o3.spherical_harmonics(l, x, normalize=True)  # (n_batch x n_res x n_channel x n_points x 2l+1)
                r = R(x)  # (n_batch x n_res x n_channel x n_points)
                r = r.unsqueeze(-1)

                ret = ret + torch.sum(c_nl * y * r, dim=-1) # (n_batch x n_res x n_channel x n_points)
            return ret
        return rho


if __name__ == "__main__":
    """ Perform a forward and backward Zernike transform and plot on a grid """
    import matplotlib.pyplot as plt

    zt = ZernikeTransform(5, 1)

    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    z = np.linspace(-1, 1, 20)
    x, y, z = np.meshgrid(x, y, z)
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x*x + y*y))

    r_filter = (r < 1)
    r = r[r_filter]
    theta = theta[r_filter]
    phi = phi[r_filter]
    x = x[r_filter]
    y = y[r_filter]
    z = z[r_filter]

    xyz = torch.from_numpy(np.array([x, y, z])).T.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor('darkgrey')

    dirac = torch.tensor([[0.5, 0.5, 0.5],
                          [-0.5, -0.5, -0.5]],
                         requires_grad=True).view([1, 1, 2, 3])
    Z = zt.forward_transform(dirac, torch.ones(dirac.shape[:-1]))
    rho = zt.back_transform(Z)

    print(xyz.shape)
    density = rho(xyz)
    print(density)
    density = density.numpy(force=True)
    p = ax.scatter(x, y, z, c=density, alpha=0.5)
    fig.colorbar(p)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
