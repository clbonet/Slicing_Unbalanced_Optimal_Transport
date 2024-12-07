import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils_ot_1d import emd1D, emd1D_dual, emd1D_dual_backprop
from .utils_hyperbolic import minkowski_ip, busemann_lorentz2, busemann_poincare2


def logsumexp(f, a):
    # stabilized
    assert f.dim() == a.dim()
    if f.dim() > 1:
        xm = torch.amax(f + torch.log(a),dim=1).reshape(-1,1)
        return xm + torch.log(torch.sum(torch.exp(f + torch.log(a) - xm),dim=1)).reshape(-1,1)
    else:
        xm = torch.amax(f + torch.log(a))
        return xm + torch.log(torch.sum(torch.exp(f + torch.log(a) - xm)))


def rescale_potentials(f, g, a, b, rho1, rho2):
    tau = (rho1 * rho2) / (rho1 + rho2)
    transl = tau * (logsumexp(-f / rho1, a) - logsumexp(-g / rho2, b))
    return transl


def kullback_leibler(a, b):
    return (a * (a/b +1e-12).log()).sum(dim=-1) - a.sum(dim=-1) + b.sum(dim=-1)


def sample_projections(num_features, num_projections, dummy_data, type_proj="linear", seed_proj=None):
    if seed_proj is not None:
        torch.manual_seed(seed_proj)
        
    if type_proj == "linear" or type_proj == "poincare_horo":
        projections = torch.normal(mean=torch.zeros([num_features, num_projections]), std=torch.ones([num_features, num_projections])).type(dummy_data.dtype).to(dummy_data.device)
        projections = F.normalize(projections, p=2, dim=0)
    elif type_proj == "lorentz_geod" or type_proj == "lorentz_horo":
        vs = np.random.normal(size=(num_projections, num_features-1))
        vs = F.normalize(torch.from_numpy(vs), p=2, dim=-1).type(dummy_data.dtype).to(dummy_data.device)
        projections = F.pad(vs, (1,0))
        
    return projections


def project_support(x, y, projections, type_proj="linear"):
    if type_proj == "linear":
        x_proj = (x @ projections).T
        y_proj = (y @ projections).T
        
    elif type_proj == "lorentz_geod":
        n_proj, d = projections.shape

        x0 = torch.zeros((1,d), device=x.device)
        x0[0,0] = 1
        
        ip_x0_x = minkowski_ip(x0, x)
        ip_v_x = minkowski_ip(projections, x)

        ip_x0_y = minkowski_ip(x0, y)
        ip_v_y = minkowski_ip(projections, y)

        x_proj = torch.arctanh(-ip_v_x/ip_x0_x).reshape(-1, n_proj).T
        y_proj = torch.arctanh(-ip_v_y/ip_x0_y).reshape(-1, n_proj).T
        
    elif type_proj == "lorentz_horo":
        n_proj, d = projections.shape

        x0 = torch.zeros((1,d), device=x.device)
        x0[0,0] = 1
        
        x_proj = busemann_lorentz2(projections, x, x0).reshape(-1, n_proj).T
        y_proj = busemann_lorentz2(projections, y, x0).reshape(-1, n_proj).T
        
    elif type_proj == "poincare_horo":
        d, n_proj = projections.shape

        x_proj = busemann_poincare2(projections.T, x).reshape(-1, n_proj).T
        y_proj = busemann_poincare2(projections.T, y).reshape(-1, n_proj).T
        
    return x_proj, y_proj


def sort_support(x_proj):
    x_sorted, x_sorter = torch.sort(x_proj, -1)
    x_rev_sort = torch.argsort(x_sorter, dim=-1)
    return x_sorted, x_sorter, x_rev_sort


def sample_project_sort_data(x, y, num_projections, type_proj="linear", seed_proj=None, projections=None):
    num_features = x.shape[1] # data dim

    # Random projection directions, shape (num_features, num_projections)
    if projections is None:
        projections = sample_projections(num_features, num_projections, dummy_data=x, seed_proj=seed_proj, type_proj=type_proj)

    # 2 ---- Project samples along directions and sort
    x_proj, y_proj = project_support(x, y, projections, type_proj)
    x_sorted, x_sorter, x_rev_sort = sort_support(x_proj)
    y_sorted, y_sorter, y_rev_sort = sort_support(y_proj)
    return x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections

    

def sliced_unbalanced_ot(a, b, x, y, p, num_projections, rho1, rho2=None, niter=10, mode='backprop',
                         seed_proj=None, type_proj="linear", projections=None):
    """
        Compute SUOT

        Parameters
        ----------
        a: tensor, shape (n_samples_a,), weights in the source domain
        b: tensor, shape (n_samples_b,), weights in the target domain
        x: tensor, shape (n_samples_a, d), samples in the source domain
        y: tensor, shape (n_samples_b, d), samples in the target domain
        p: float, power
        num_projections: int, number of projections
        rho1: float, first marginal relaxation term
        rho2: float, second marginal relaxation term (default = rho1)
        niter: int, number of Frank-Wolfe algorithm
        mode: "backprop" or "icdf", how to compute the potentials
        seed_proj
        type_proj: "linear", "lorentz_geod", "lorentz_horo" or "poincare_horo": Euclidean or hyperbolic projection
        projections: shape (d, num_projections), by default None and sample projections
    """
    if rho2 is None:
        rho2 = rho1
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj, projections)
    a = a[..., x_sorter]
    b = b[..., y_sorter]

    # 3 ----- Prepare and start FW

    # Initialize potentials
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for k in range(niter):
        # Output FW descent direction
        transl = rescale_potentials(f, g, a, b, rho1, rho2)

        # translate potentials
        f = f + transl
        g = g - transl
        # update measures
        A = a * torch.exp(-f / rho1)
        B = b * torch.exp(-g / rho2)
        # solve for new potentials
        if mode == 'icdf':
            fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        if mode == 'backprop':
            fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        # default step for FW
        t = 2. / (2. + k)

        f = f + t * (fd - f)
        g = g + t * (gd - g)

    # 4 ----- We are done. Get me out of here !
    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2)
    f, g = f + transl, g - transl
    A, B = a * torch.exp(-f / rho1), b * torch.exp(-g / rho2)
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    loss = loss + rho1 * torch.mean(kullback_leibler(A, a)) + rho2 * torch.mean(kullback_leibler(B, b))

    # Reverse sort potentials and measures w.r.t order not sample (not sorted)
    f, g = torch.gather(f, 1, x_rev_sort), torch.gather(g, 1, y_rev_sort)
    A, B = torch.gather(A, 1, x_rev_sort), torch.gather(B, 1, y_rev_sort)
    
    return loss, f, g, A, B, projections



def unbalanced_sliced_ot(a, b, x, y, p, num_projections, rho1, rho2=None, niter=10,
                         mode='backprop', stochastic_proj=False, seed_proj=None, type_proj="linear", projections=None):
    """
        Compute USOT

        Parameters
        ----------
        a: tensor, shape (n_samples_a,), weights in the source domain
        b: tensor, shape (n_samples_b,), weights in the target domain
        x: tensor, shape (n_samples_a, d), samples in the source domain
        y: tensor, shape (n_samples_b, d), samples in the target domain
        p: float, power
        num_projections: int, number of projections
        rho1: float, first marginal relaxation term
        rho2: float, second marginal relaxation term (default =rho1)
        niter: int, number of Frank-Wolfe algorithm
        mode: "backprop" or "icdf", how to compute the potentials
        seed_proj
        type_proj: "linear", "lorentz_geod", "lorentz_horo" or "poincare_horo": Euclidean or hyperbolic projection
        projections: shape (d, num_projections), by default None and sample projections
    """
    if rho2 is None:
        rho2 = rho1
        
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    if not stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj, projections)

    # 3 ----- Prepare and start FW

    # Initialize potentials - WARNING: They correspond to non-sorted samples
    f = torch.zeros(x.shape[0], dtype=a.dtype, device=a.device)
    g = torch.zeros(y.shape[0], dtype=a.dtype, device=a.device)

    for k in range(niter):
        # Output FW descent direction
        # translate potentials
        transl = rescale_potentials(f, g, a, b, rho1, rho2)
        f = f + transl
        g = g - transl

        # If stochastic version then sample new directions and re-sort data
        if stochastic_proj:
            x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj)

        # update measures
        A = (a * torch.exp(-f / rho1))[..., x_sorter]
        B = (b * torch.exp(-g / rho2))[..., y_sorter]
        
        # solve for new potentials
        if mode == 'icdf':
            fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        if mode == 'backprop':
            fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        # default step for FW
        t = 2. / (2. + k)
        f = f + t * (torch.mean(torch.gather(fd, 1, x_rev_sort), dim=0) - f)
        g = g + t * (torch.mean(torch.gather(gd, 1, y_rev_sort), dim=0) - g)

    # 4 ----- We are done. Get me out of here !
    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2)
    f, g = f + transl, g - transl
    A, B = (a * torch.exp(-f / rho1))[..., x_sorter], (b * torch.exp(-g / rho2))[..., y_sorter]
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    A, B = a * torch.exp(-f / rho1), b * torch.exp(-g / rho2)
    loss = loss + rho1 * kullback_leibler(A, a) + rho2 * kullback_leibler(B, b)
    
    return loss, f, g, A, B, projections
    

def sliced_ot(a, b, x, y, p, num_projections, niter=10, mode='backprop', stochastic_proj=False, seed_proj=None, type_proj="linear"):
    """
        Compute SOT

        Parameters
        ----------
        a: tensor, shape (n_samples_a,), weights in the source domain
        b: tensor, shape (n_samples_b,), weights in the target domain
        x: tensor, shape (n_samples_a, d), samples in the source domain
        y: tensor, shape (n_samples_b, d), samples in the target domain
        p: float, power
        num_projections: int, number of projections
        niter: int, number of Frank-Wolfe algorithm
        mode: "backprop" or "icdf", how to compute the potentials
        seed_proj
        type_proj: "linear", "lorentz_geod", "lorentz_horo" or "poincare_horo": Euclidean or hyperbolic projection
    """
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    if not stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj)

    # 3 ----- Prepare and start FW

    # Initialize potentials - WARNING: They correspond to non-sorted samples
    f = torch.zeros(x.shape[0], dtype=a.dtype, device=a.device)
    g = torch.zeros(y.shape[0], dtype=a.dtype, device=a.device)

    # Output FW descent direction

    # If stochastic version then sample new directions and re-sort data
    if stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj)

    # update measures
    A = a[..., x_sorter]
    B = b[..., y_sorter]
    
    # solve for new potentials
    if mode == 'icdf':
        fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
    if mode == 'backprop':
        fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
    # default step for FW
    f = torch.mean(torch.gather(fd, 1, x_rev_sort), dim=0)
    g = torch.mean(torch.gather(gd, 1, y_rev_sort), dim=0)

    # 4 ----- We are done. Get me out of here !
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    
    return loss, f, g, projections
