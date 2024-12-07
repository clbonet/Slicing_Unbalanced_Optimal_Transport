import pytest
import torch
from sliceduot.utils_ot_1d import emd1D, emd1D_dual, emd1D_dual_backprop
from sliceduot.utils_sampling import generate_measure

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


@pytest.mark.parametrize("p", [1, 1.5, 2])
def test_sliced_emd_nobug(p):
    nproj, n, m = 3, 4, 5
    a, x = generate_measure(nproj, n, slice=True)
    b, y = generate_measure(nproj, m, slice=True)
    for u_weights in [None, a, a[0]]:
        for v_weights in [None, b, b[0]]:
            emd1D(x, y, u_weights=u_weights, v_weights=v_weights, p=p, require_sort=True)
            emd1D_dual(x, y, u_weights=u_weights, v_weights=v_weights,p=p, require_sort=True)
            emd1D_dual_backprop(x, y, u_weights=u_weights, v_weights=v_weights,p=p, require_sort=True)


@pytest.mark.parametrize("p", [1, 1.5, 2])
def test_sliced_emd_definite(p):
    nproj, n = 3, 4
    a, x = generate_measure(nproj, n)

    for u_weights in [None, a, a[0]]:
        loss = emd1D(x, x, u_weights=u_weights, v_weights=u_weights,p=p, require_sort=True)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
        f, g, loss = emd1D_dual(x, x, u_weights=u_weights, v_weights=u_weights, p=p, require_sort=True)
        assert torch.allclose(f, torch.zeros_like(f), atol=1e-6)
        assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
        f, g, loss = emd1D_dual_backprop(x, x, u_weights=u_weights, v_weights=u_weights, p=p, require_sort=True)
        assert torch.allclose(f, torch.zeros_like(f), atol=1e-6)
        assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


@pytest.mark.parametrize("p", [1, 1.5, 2])
def test_consistency_emd_loss(p):
    nproj, n, m = 3, 4, 5
    a, x = generate_measure(nproj, n)
    b, y = generate_measure(nproj, m)

    # Sort support
    x, _ = torch.sort(x, -1)
    y, _ = torch.sort(y, -1)

    loss = emd1D(x, y, u_weights=a, v_weights=b, p=p, require_sort=False)
    _, _, lossn, lossp = numpy_sliced_emd(a, b, x, y, p)
    print("LOSS NUMPY = \n", lossn)
    print("LOSS POT = \n", lossp)
    print("LOSS QUANTILE = \n", loss)
    assert torch.allclose(loss, lossn, atol=1e-4)
    assert torch.allclose(loss, lossp, atol=1e-4)

@pytest.mark.parametrize("p", [1, 2])
def test_consistency_emd_dual_potentials(p):
    nproj, n, m = 3, 4, 5
    a, x = generate_measure(nproj, n)
    b, y = generate_measure(nproj, m)

    # Sort support
    x, _ = torch.sort(x, -1)
    y, _ = torch.sort(y, -1)
    print("SUPPORT X = \n", x)
    print("SUPPORT Y = \n", y)

    f, g, loss = emd1D_dual(x, y, u_weights=a, v_weights=b, p=p, require_sort=False)
    fd, gd, lossd = emd1D_dual_backprop(x, y, u_weights=a, v_weights=b, p=p, require_sort=False)
    fn, gn, lossn, lossp = numpy_sliced_emd(a, b, x, y, p)
    # gd = gd + fd[:,0][:,None]
    # fd = fd - fd[:,0][:,None]
    print("LOSS NUMPY = \n", lossn)
    print("LOSS POT = \n", lossp)
    print("LOSS QUANTILE = \n", loss)
    print("Quantile = \n", f)
    print("Backprop = \n", fd)
    print("Numpy = \n", fn)
    print("DIFFERENCE = \n", fd-fn)
    print("\n\n")
    print("Backprop G = \n", gd)
    print("Numpy G = \n", gn)
    print("DIFFERENCE G = \n", gd-gn)
    assert torch.allclose(loss, lossn, atol=1e-6)
    # assert torch.allclose(torch.var(fn-f), torch.zeros([1]), atol=1e-6)
    assert torch.allclose(torch.var(fn-fd, dim=1), torch.zeros([nproj]), atol=1e-6)
    # assert torch.allclose(torch.var(fd-f), torch.zeros([1]), atol=1e-6)
    # assert torch.allclose(torch.var(gn-g), torch.zeros([1]), atol=1e-6)
    assert torch.allclose(torch.var(gn-gd, dim=1), torch.zeros([nproj]), atol=1e-6)
    # assert torch.allclose(torch.var(gd-g), torch.zeros([1]), atol=1e-6)

# @pytest.mark.parametrize("p", [1, 1.5, 2])
# def test_sliced_emd_sensitive_same_sample(p):
#     nproj, n, m = 3, 4, 5
#     _, x = generate_measure(nproj, n)
#     _, y = generate_measure(nproj, m)
#     s = torch.distributions.exponential.Exponential(1.0)
#     a = s.sample(torch.Size([n]))
#     a = a / a.sum()
#     b = s.sample(torch.Size([m]))
#     b = b / b.sum()
#     yd = torch.hstack((y,y))
#     bd = 0.5 * torch.hstack((b,b))
#     print("y", y, y.size())
#     print("yd", yd, yd.size())
#     print("b", b, b.size())
#     print("bd", bd, bd.size())

#     loss = emd1D(x, y, u_weights=None, v_weights=None,p=p, require_sort=True)
#     lossd = emd1D(x, yd, u_weights=None, v_weights=None,p=p, require_sort=True)
#     assert torch.allclose(loss, lossd, atol=1e-6)

#     loss = emd1D(x, y, u_weights=a, v_weights=b, p=p, require_sort=True)
#     lossd = emd1D(x, yd, u_weights=a, v_weights=bd, p=p, require_sort=True)
#     assert torch.allclose(loss, lossd, atol=1e-6)

#     f, g, loss = emd1D_dual(x, y, u_weights=None, v_weights=None,p=p, require_sort=True)
#     fd, gd, lossd = emd1D_dual(x, yd, u_weights=None, v_weights=None,p=p, require_sort=True)
#     print("f", f)
#     print("fd", fd)
#     assert torch.allclose(loss, lossd, atol=1e-6)
#     assert torch.allclose(f, fd, atol=1e-6)
#     assert torch.allclose(torch.hstack((g,g)), gd)

#     f, g, loss = emd1D_dual(x, y, u_weights=a, v_weights=b,p=p, require_sort=True)
#     fd, gd, lossd = emd1D_dual(x, yd, u_weights=a, v_weights=bd, p=p, require_sort=True)
#     assert torch.allclose(loss, lossd, atol=1e-6)
#     assert torch.allclose(f, fd)
#     assert torch.allclose(torch.hstack((g,g)), gd)




# def test_sliced_emd_sensitive_zero_mass():
#     # TODO