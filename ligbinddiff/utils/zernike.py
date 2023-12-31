""" Helper functions for zernike transforms in pytorch

Parts adapted from https://github.com/StatPhysBio/protein_holography/blob/main/protein_holography/hologram/hologram.py """

from e3nn import o3
import numpy as np
import torch
import scipy.special


# def _log_binom(x, y):
#     """ Compute the generalized binomial coefficient """
#     x = torch.as_tensor(x)
#     y = torch.as_tensor(y)
#     log_fac_x = torch.lgamma(x+1)  # log n!
#     log_fac_y = torch.lgamma(y+1)  # log k!
#     log_fac_xmy = torch.lgamma(x-y+1)  # log (n-k)!
#     return log_fac_x - log_fac_y - log_fac_xmy  # binom(x, y)
#
# def _log_poch(a, n):
#     a = torch.as_tensor(a)
#     n = torch.as_tensor(n)
#     log_fac_a = torch.lgamma(a+1)
#     log_fac_amn = torch.lgamma(a-n+1)
#     return log_fac_a - log_fac_amn
#
# # see https://en.wikipedia.org/wiki/Hypergeometric_function
# def hyp2f1(a, b, c, z):
#     assert int(a) == a, "a must be of integer type"
#     assert a <= 0, "a must be a nonnegative integer"
#
#     ret = torch.tensor([0], device=z.device)
#     m = -int(a)
#     for n in range(m):
#         ret = ret + (-1)**n * torch.exp(
#             _log_binom(m, n) + _log_poch(b, n) - _log_poch(c, n)
#             ) * torch.pow(z, n)
#
#     return ret


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

        # hack for now
        E = E.to(F.device)

        # print("radial", n, l, E.isnan().any(), F.isnan().any())
        # assemble coefficients
        radial_out = A * B * C * E * F

        return radial_out  # (n_batch x n_res x n_points)
    return radial


class ZernikeTransform:
    """ A cache for Zernike radial polynomials """
    def __init__(self, n_max=None, r_scale=None, compact=True):
        self.cache = None
        self.compact = compact
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
        for n in range(n_max+1):
            for l in range(n+1):
                if (n - l) % 2 == 1 and self.compact:
                    continue
                self.cache[(n, l)] = zernike_radial(n, l, r_scale)

    def forward_transform(self, points, center, points_mask, point_value=None):
        """ Compute the zernike polynomial expansion coefficients for a point cloud.
        The points are treated as a sum of Dirac distributions.

        Args
        -----
        points : torch.Tensor ([n_batch x] n_res x n_points x 3)
            point cloud to encode
        center : torch.Tensor ([n_batch x] n_res x 3)
            the origin around which to encode each point
        points_mask : torch.Tensor ([n_batch x] n_res x n_points)
            point cloud mask
        point_value: None, int, or torch.Tensor ([n_batch x n_res x] n_channel x n_points)
            scalr value per point, can be channeled
        """

        points = points - center.unsqueeze(-2)

        if point_value is None:
            point_value = 1
            n_channels = 1
        elif len(point_value.shape) == 2:
            n_channels = point_value.shape[0]
            n_res = points.shape[-3]

            point_value = point_value.unsqueeze(-1)
            point_value = point_value.unsqueeze(0)
            point_value = point_value.expand(n_res, -1, -1, -1)
        else:
            n_channels = point_value.shape[1]
            assert len(points.shape) == len(point_value.shape)

        points_mask = points_mask.unsqueeze(-2).expand(-1, n_channels, -1)  # add channel

        Z = {}
        for n in range(self.n_max+1):
            for l in range(n+1):
                # we retain the zeros since they're used in se3 transformers
                if (n-l) % 2 == 1:
                    if not self.compact:
                        Z[(n, l)] = torch.zeros(list(points_mask.shape[:-1]) + [2*l+1])
                    continue

                R = self.get(n, l)
                y = o3.spherical_harmonics(l, points, normalize=True)  # ([n_batch x] n_res x n_points x 2l+1)
                r = R(points)  # ([n_batch x] n_res x n_points)

                # add channel
                y = y.unsqueeze(-3)  # ([n_batch x] n_res x n_channel x n_points x 2l+1)
                r = r.unsqueeze(-2)  # ([n_batch x] n_res x n_channel x n_points)
                # fix shapes for broadcasting
                r = r.unsqueeze(-1)  # ([n_batch x] n_res x n_channel x n_points x 1)
                y = y.expand(-1, n_channels, -1, -1)
                r = r.expand(-1, n_channels, -1, -1)
                c_nl = (y * r * point_value)  # [n_batch x] n_res x n_channel x n_points x 2l+1
                c_nl = c_nl * points_mask[..., None]

                Z[(n, l)] = c_nl.sum(-2)
        return Z  # Dict[Tuple[int, int], Tensor[n_batch x n_res x n_channel x 2l+1]]

    def back_transform(self, Z, X_cb, device='cpu'):
        """ Reverse a zernike transform

        Args
        -----
        Z : Dict
            a dictionary of zernike polynomial expansion coefficients

        Returns
        -------
        rho : func: R^3 -> R
            density function
        X_cb : torch.Tensor (n_res x 3)
        """
        X_cb_mask = ~(X_cb.isnan()).any(dim=-1)

        def rho(x):
            """ Density function
            Args
            ----
            x : torch.Tensor (... x 3)
            """
            # compute the queried point relative to each C_alpha
            x_rel = x.unsqueeze(-2).to(device) - X_cb[X_cb_mask].to(device)  # (... x n_res x 3)
            # for numerical stability reasons, ignore all contributions
            # where the query is outside the ball of support per residue
            x_rel_mag = torch.linalg.vector_norm(x_rel, dim=-1)
            x_rel_mask = (x_rel_mag > self.r_scale)  # (... x n_res)

            ret = torch.tensor([0.], device=device)
            for (n, l), c_nl in Z.items():
                if (n-l) % 2 == 1:
                    continue

                c_nl = c_nl.to(device)

                R = self.get(n, l)
                y = o3.spherical_harmonics(l, x_rel, normalize=True)  # (... x n_res x 2l+1)
                r = R(x_rel)  # (... x n_res)
                r = r.unsqueeze(-1)  # (... x n_res x 1)

                y = y.unsqueeze(-2)  # (... x n_res x 1 x 2l+1)
                r = r.unsqueeze(-2)  # (... x n_res x 1 x 1)
                c_nl_contrib = y * r  # (... x n_res x 1 x 2l+1)
                c_nl_contrib = c_nl[X_cb_mask] * c_nl_contrib  # (... x n_res x n_channel x 2l+1)
                c_nl_contrib[x_rel_mask] = 0
                # print(c_nl_contrib.shape)

                ret = ret + c_nl_contrib.sum(dim=-1) # (...)
            return ret.max(dim=-2)[0]
        return rho

    def reswise_back_transform(self, Z, device='cpu'):
        """ Reverse a zernike transform per residue

        Args
        -----
        Z : Dict
            a dictionary of zernike polynomial expansion coefficients

        Returns
        -------
        rho : func: R^3 -> R
            density function
        """
        def rho(x_rel, x_mask, channel_mask):
            """ Density function
            Args
            ----
            x_rel : torch.Tensor (... x n_res x n_atom x 3)

            x_mask : torch.Tensor (... x n_res x n_atom)

            channel_mask : torch.Tensor (n_atom x n_channel)
            """
            ret = torch.tensor([0.], device=device)
            x_mask = x_mask.to(x_rel.device)
            channel_mask = channel_mask.to(x_rel.device)
            for (n, l), c_nl in Z.items():
                if (n-l) % 2 == 1:
                    continue

                c_nl = c_nl.to(device)

                R = self.get(n, l)
                y = o3.spherical_harmonics(l, x_rel, normalize=True)  # (... x n_res x n_atom x 2l+1)
                r = R(x_rel)  # (... x n_res x n_atom)
                r = r.unsqueeze(-1)  # (... x n_res x n_atom x 1)

                y = y.unsqueeze(-2)  # (... x n_res x n_atom x 1 x 2l+1)
                r = r.unsqueeze(-2)  # (... x n_res x n_atom x 1 x 1)
                c_nl_contrib = y * r  # (... x n_res x n_atom x 1 x 2l+1)
                c_nl_contrib[x_mask] = 0
                c_nl = c_nl.unsqueeze(-3)

                c_nl_contrib = c_nl * c_nl_contrib  # (... x n_res x n_atom x n_channel x 2l+1)
                c_nl_contrib[..., channel_mask, :] = 0

                ret = ret + c_nl_contrib.sum(dim=-1) # (... x n_res x n_atom x n_channel)
            return ret
        return rho


if __name__ == "__main__":
    """ Perform a forward and backward Zernike transform and plot on a grid """
    import matplotlib.pyplot as plt

    zt = ZernikeTransform(5, 1)

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)
    x, y, z = np.meshgrid(x, y, z)

    center = np.array([0., 0., 0.])
    x_center = x - center[0]
    y_center = y - center[1]
    z_center = z - center[2]

    r = np.sqrt(x_center*x_center + y_center*y_center + z_center*z_center)
    r_filter = (r < 1)
    x = x[r_filter]
    y = y[r_filter]
    z = z[r_filter]

    xyz = torch.from_numpy(np.array([x, y, z])).T

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor('darkgrey')

    dirac = torch.tensor([
        [[0.5, 0.5, 0.5],
         [-0.5, -0.5, -0.5]]
    ], requires_grad=True)
    Z = zt.forward_transform(dirac, torch.from_numpy(center), torch.zeros(dirac.shape[:-1]).bool())
    rho = zt.back_transform(Z, X_cb=torch.from_numpy(center))

    print(xyz.shape)
    density = rho(xyz)
    print(density.shape)
    density = density.numpy(force=True)
    p = ax.scatter(x, y, z, c=density, alpha=0.5)
    fig.colorbar(p)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    # x = np.linspace(0, 2, 10)
    # y = np.linspace(0, 2, 10)
    # z = np.linspace(0, 2, 10)
    # x, y, z = np.meshgrid(x, y, z)

    # center = np.array([1., 1., 1.])
    # x_center = x - center[0]
    # y_center = y - center[1]
    # z_center = z - center[2]

    # r = np.sqrt(x_center*x_center + y_center*y_center + z_center*z_center)
    # r_filter = (r < 1)
    # x = x[r_filter]
    # y = y[r_filter]
    # z = z[r_filter]

    # xyz = torch.from_numpy(np.array([x, y, z])).T

    # fig = plt.figure(figsize=plt.figaspect(1.))
    # ax = fig.add_subplot(projection='3d')
    # ax.set_facecolor('darkgrey')

    # dirac = torch.tensor([
    #     [[1.5, 1.5, 1.5],
    #      [0.5, 0.5, 0.5]]
    # ], requires_grad=True)
    # Z = zt.forward_transform(dirac, torch.from_numpy(center), torch.zeros(dirac.shape[:-1]).bool())
    # rho = zt.back_transform(Z, X_ca=torch.from_numpy(center))

    # print(xyz.shape)
    # density = rho(xyz)
    # print(density.shape)
    # density = density.numpy(force=True)
    # p = ax.scatter(x, y, z, c=density, alpha=0.5)
    # fig.colorbar(p)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()
