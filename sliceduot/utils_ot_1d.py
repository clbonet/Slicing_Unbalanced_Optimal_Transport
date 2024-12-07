import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sliceduot.utils_sampling import generate_measure


def cost_function(p):
    """Outputs the cost C(x-y) function used to compare the inverse cdf
    WARNING: The cost C must be a convex function.

    Args:
        p (int): Exponent of the cost C(x) = |x|^p

    Returns:
        function: torch function which takes torch.Tensor as input and output.
    """
    if p == 1:
        return torch.abs
    if p == 2:
        return torch.square
    else:
        def cost(x):
            return torch.pow(torch.abs(x), p)
        return cost



def emd1D(u_values, v_values, u_weights=None, v_weights=None,p=2, require_sort=True):
    """computes sliced-wise the p-th norm between the inverse cdf of two measures

    Args:
        u_values (torch.Tensor of size [Proj, N]): support of first measures
        v_values (torch.Tensor of size [Proj, M]): support of second measures
        u_weights (torch.Tensor of size [Proj, N]): weights of first measures. Defaults to None for uniform weights.
        v_weights (torch.Tensor of size [Proj, M]): weights of second measures. Defaults to None for uniform weights.
        p (int, optional): Exponent of cost C(x)=|x|^p. Defaults to 2.
        require_sort (bool, optional): Ask whether support must be sorted or not. Defaults to True if inputs are already sorted.

    Returns:
        loss (torch.Tensor of size [Proj]): univariate OT loss between the [Proj] pairs of measures.
    """
    proj = u_values.shape[0]
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    # Init weights or broadcast if necessary
    if u_weights is None:
        u_weights = torch.full((proj,n), 1/n, dtype=dtype, device=device)
    elif u_weights.dim() == 1:
        u_weights = u_weights.repeat(u_values.shape[0], 1)
    assert (u_weights.dim()) == 2 and (u_values.size()== u_weights.size())


    if v_weights is None:
        v_weights = torch.full((proj,m), 1/m, dtype=dtype, device=device)
    elif v_weights.dim() == 1:
        v_weights = v_weights.repeat(v_values.shape[0], 1)
    assert (v_weights.dim()) == 2 and (v_values.size()== v_weights.size())

    # Sort w.r.t. support if not already done
    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = torch.gather(u_weights, -1, u_sorter)
        v_weights = torch.gather(v_weights, -1, v_sorter)
        # u_weights = u_weights[..., u_sorter]
        # v_weights = v_weights[..., v_sorter]
   
    u_cdf = torch.clamp(torch.cumsum(u_weights, -1), max=1.)
    v_cdf = torch.clamp(torch.cumsum(v_weights, -1), max=1.)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    cost = cost_function(p)
    loss = torch.sum(delta * cost(u_icdf - v_icdf), axis=-1)
    return loss



def emd1D_dual(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    
    proj = u_values.shape[0]
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    # Init weights or broadcast if necessary
    if u_weights is None:
        u_weights = torch.full((proj,n), 1/n, dtype=dtype, device=device)
    elif u_weights.dim() == 1:
        u_weights = u_weights.repeat(u_values.shape[0], 1)
    assert (u_weights.dim()) == 2 and (u_values.size()== u_weights.size())


    if v_weights is None:
        v_weights = torch.full((proj,m), 1/m, dtype=dtype, device=device)
    elif v_weights.dim() == 1:
        v_weights = v_weights.repeat(v_values.shape[0], 1)
    assert (v_weights.dim()) == 2 and (v_values.size()== v_weights.size())

    # Sort w.r.t. support if not already done
    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = torch.gather(u_weights, 1, u_sorter)
        v_weights = torch.gather(v_weights, 1, v_sorter)

    # eps trick to have strictly increasing cdf and avoid zero mass issues
    eps=1e-12
    u_cdf = torch.cumsum(u_weights + eps, -1) - eps
    v_cdf = torch.cumsum(v_weights + eps, -1) - eps

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis).clip(0, n-1)
    v_index = torch.searchsorted(v_cdf, cdf_axis).clip(0, m-1)

    u_icdf = torch.gather(u_values, -1, u_index)
    v_icdf = torch.gather(v_values, -1, v_index)

    cost = cost_function(p)
    diff_dist = cost(u_icdf - v_icdf)
    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    loss = torch.sum((cdf_axis[..., 1:] - cdf_axis[..., :-1]) * diff_dist, axis=-1)
    
    mask_u = u_index[...,1:]-u_index[...,:-1]
    mask_u = torch.nn.functional.pad(mask_u, (1, 0))
    mask_v = v_index[...,1:]-v_index[...,:-1]
    mask_v = torch.nn.functional.pad(mask_v, (1, 0))

    c1 = torch.where((mask_u[...,:-1]+mask_u[...,1:])>1,-1,0)
    c1 = torch.cumsum(c1*diff_dist[...,:-1],dim=-1)
    c1 = torch.nn.functional.pad(c1, (1, 0))

    c2 = torch.where((mask_v[...,:-1]+mask_v[...,1:])>1,-1,0)
    c2 = torch.cumsum(c2*diff_dist[...,:-1],dim=-1)
    c2 = torch.nn.functional.pad(c2, (1, 0))
    
    masked_u_dist = mask_u*diff_dist
    masked_v_dist = mask_v*diff_dist

    T = torch.cumsum(masked_u_dist-masked_v_dist,dim=-1) + c1  - c2
    tmp = mask_u.clone() # avoid in-place problem
    tmp[...,0]=1
    f = torch.masked_select(T, tmp.bool()).reshape_as(u_values)
    f[...,0]=0
    tmp = mask_v.clone() # avoid in-place problem
    tmp[...,0]=1
    g = -torch.masked_select(T, tmp.bool()).reshape_as(v_values) # TODO: Apparently buggy line (v_values/mask format unstable)
    return f, g, loss


def emd1D_dual_backprop(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]
    if u_weights is None:
        mu_1 = torch.full((u_values.shape[0], u_values.shape[1]), 1/u_values.shape[1], dtype=u_values.dtype, device=u_values.device)
    elif u_weights.dim() == 1:
        mu_1 = u_weights.repeat(u_values.shape[0], 1).clone().detach()
    else:
        mu_1 = u_weights.clone().detach()
    assert u_values.dim() == mu_1.dim()

    if v_weights is None:
        mu_2 = torch.full((v_values.shape[0], v_values.shape[1]), 1/v_values.shape[1], dtype=v_values.dtype, device=v_values.device)
    elif v_weights.dim() == 1:
        mu_2 = v_weights.repeat(v_values.shape[0], 1).clone().detach()
    else:
        mu_2 = v_weights.clone().detach()
    assert v_values.dim() == mu_2.dim()
    
    mu_1.requires_grad_(True)
    mu_2.requires_grad_(True)
    value = emd1D(u_values, v_values, u_weights=mu_1, v_weights=mu_2, p=p, require_sort=require_sort).sum()
    value.backward()

    return mu_1.grad, mu_2.grad, value # value can not be backward anymore


if __name__ == '__main__':
    import numpy as np
    import numba
    from numba import jit
    from ot import emd2_1d
    @jit(nopython=True)
    def solve_ot(a, b, x, y, p):
        """Computes the 1D Optimal Transport between two histograms.
        _Important: one should have np.sum(a)=np.sum(b)._
        _Important:_ x and y needs to be sorted.
        Parameters
        ----------
        a: vector of length n with positive entries
        b: vector of length m with positive entries
        x: vector of real of length n
        y: vector of real of length m
        p: real, should >= 1
        Returns
        ----------
        I: vector of length q=n+m-1 of increasing integer in {0,...,n-1}
        J: vector of length q of increasing integer in {0,...,m-1}
        P: vector of length q of positive values of length q
        f: dual vector of length n
        g: dual vector of length m
        cost: (dual) OT cost
            sum a_i f_i + sum_j b_j f_j
            It should be equal to the primal cost
            = sum_k |x(i)-y(j)|^p where i=I(k), j=J(k)
        """
        n = len(a)
        m = len(b)
        q = m + n - 1
        a1 = a.copy()
        b1 = b.copy()
        I = np.zeros(q).astype(numba.int64)
        J = np.zeros(q).astype(numba.int64)
        P = np.zeros(q)
        f = np.zeros(n)
        g = np.zeros(m)
        g[0] = np.abs(x[0] - y[0]) ** p
        for k in range(q - 1):
            i = I[k]
            j = J[k]
            if (a1[i] < b1[j]) and (i < n - 1):
                I[k + 1] = i + 1
                J[k + 1] = j
                f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
            elif (a1[i] > b1[j]) and (j < m - 1):
                I[k + 1] = i
                J[k + 1] = j + 1
                g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
            elif i == n - 1:
                I[k + 1] = i
                J[k + 1] = j + 1
                g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
            elif j == m - 1:
                I[k + 1] = i + 1
                J[k + 1] = j
                f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
            t = min(a1[i], b1[j])
            P[k] = t
            a1[i] = a1[i] - t
            b1[j] = b1[j] - t
        P[k + 1] = max(a1[-1], b1[-1])  # remaining mass
        cost = np.sum(f * a) + np.sum(g * b)
        return I, J, P, f, g, cost


    def numpy_sliced_emd(a, b, x, y, p):
        A, B = a.data.numpy().astype('float64'), b.data.numpy().astype('float64')
        X, Y = x.data.numpy().astype('float64'), y.data.numpy().astype('float64')
        f, g = np.zeros_like(x), np.zeros_like(y)
        loss = np.zeros_like(f[:,0])
        loss_pot = np.zeros_like(f[:,0])
        assert X.shape[0] == Y.shape[0]
        for i in range(X.shape[0]):
            _, _, _, F, G, L = solve_ot(A[i], B[i], X[i], Y[i], p)
            L2 = emd2_1d(X[i], Y[i], A[i], B[i], metric='minkowski', p=p)
            f[i], g[i], loss[i], loss_pot[i] = F, G, L, L2
        return torch.from_numpy(f), torch.from_numpy(g), torch.from_numpy(loss), torch.from_numpy(loss_pot)

    # n, dim, nproj = 3, 4, 5
    # rho1, rho2 = 1e-1, 1e0
    # p = 1.5
    # a, x = generate_measure(n, dim)
    # loss, f, g, A, B, _ = sliced_unbalanced_ot(a, a, x, x, p=p, num_projections=nproj, rho1=rho1, rho2=rho2, niter=10)
    nproj, n, m = 3, 4, 5
    p = 2
    a, x = generate_measure(nproj, n)
    b, y = generate_measure(nproj, m)

    # Sort support
    x, _ = torch.sort(x, -1)
    y, _ = torch.sort(y, -1)
    print("SUPPORT X = \n", x)
    print("SUPPORT Y = \n", y)

    fn, gn, lossn, lossp = numpy_sliced_emd(a, b, x, y, p)
    print("LOSS NUMPY = \n", lossn)
    print("LOSS POT = \n", lossp)
    loss = emd1D(x, y, u_weights=a, v_weights=b, p=p, require_sort=False)
    print("LOSS QUANTILE = \n", loss)
    raise SystemExit
    f, g, _ = emd1D_dual(x, y, u_weights=a, v_weights=b, p=p, require_sort=False)
    fd, gd, _ = emd1D_dual_backprop(x, y, u_weights=a, v_weights=b, p=p, require_sort=False)
    gd = gd + fd[:,0][:,None]
    fd = fd - fd[:,0][:,None]
    print("Quantile = \n", f)
    print("Backprop = \n", fd)
    print("Numpy = \n", fn)
    raise SystemExit

    nproj, n, m = 2, 3, 4
    p = 1.5
    a, x = generate_measure(nproj, n)
    b, y = generate_measure(nproj, m)
    loss = emd1D(x, y, u_weights=a, v_weights=b,p=p, require_sort=True)
    fd, gd, lossd = emd1D_dual_backprop(x, y, u_weights=a, v_weights=b,p=p, require_sort=True)
    raise SystemExit
    # s = torch.distributions.exponential.Exponential(1.0)
    # a = s.sample(torch.Size([n]))
    # a = a / a.sum()
    # b = s.sample(torch.Size([m]))
    # b = b / b.sum()
    # yd = torch.hstack((y,y))
    # bd = 0.5 * torch.hstack((b,b))
    # print("y", y, y.size())
    # print("yd", yd, yd.size())
    # print("b", b, b.size())
    # print("bd", bd, bd.size())

    # loss = emd1D(x, y, u_weights=None, v_weights=None,p=p, require_sort=True)
    # lossd = emd1D(x, yd, u_weights=None, v_weights=None,p=p, require_sort=True)
    # assert torch.allclose(loss, lossd, atol=1e-6)

    loss = emd1D(x, y, u_weights=a, v_weights=b, p=p, require_sort=True)
    lossd = emd1D(x, yd, u_weights=a, v_weights=bd, p=p, require_sort=True)
    assert torch.allclose(loss, lossd, atol=1e-6)

    f, g, loss = emd1D_dual(x, y, u_weights=None, v_weights=None,p=p, require_sort=True)
    fd, gd, lossd = emd1D_dual(x, yd, u_weights=None, v_weights=None,p=p, require_sort=True)
    fd, gd, lossd = emd1D_dual_backprop(x, yd, u_weights=a, v_weights=bd,p=p, require_sort=True)
    print("fd", fd)
    print("gd", gd)
    fd, gd, lossd = emd1D_dual_backprop(x, yd, u_weights=None, v_weights=None,p=p, require_sort=True)
    print("f", f)
    print("fd", fd)
    print("gd", gd)