import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from sliceduot.utils_sampling import generate_measure
from sliceduot.sliced_uot import sample_project_sort_data, rescale_potentials, kullback_leibler
from sliceduot.utils_ot_1d import emd1D, emd1D_dual, emd1D_dual_backprop



def USOT_store_loss(a, b, x, y, p, num_projections, rho1, rho2=None, niter=10, mode='backprop', stochastic_proj=False):
    if rho2 is None:
        rho2 = rho1
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    if not stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections)

    # 3 ----- Prepare and start FW

    # Initialize potentials - WARNING: They correspond to non-sorted samples
    f = torch.zeros(x.shape[0], dtype=a.dtype, device=a.device)
    g = torch.zeros(y.shape[0], dtype=a.dtype, device=a.device)
    log = []

    pbar = trange(niter)
    for k in pbar:
        # Output FW descent direction
        # translate potentials
        transl = rescale_potentials(f, g, a, b, rho1, rho2)
        f = f + transl
        g = g - transl

        # If stochastic version then sample new directions and re-sort data
        if stochastic_proj:
            x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections)

        # update measures
        A = (a * torch.exp(-f / rho1))[..., x_sorter]
        B = (b * torch.exp(-g / rho2))[..., y_sorter]
        
        # solve for new potentials
        if mode == 'icdf':
            fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        if mode == 'backprop':
            fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        
        loss = loss.mean() + rho1 * kullback_leibler(a * torch.exp(-f / rho1), a) + rho2 * kullback_leibler(b * torch.exp(-g / rho1), b)
        log.append(loss.data.item())
        
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
    log.append(loss.data.item())
    
    return log, f, g, A, B, projections



def SUOT_store_loss(a, b, x, y, p, num_projections, rho1, rho2=None, niter=10, mode='backprop', seed_proj=None):
    if rho2 is None:
        rho2 = rho1
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections)
    a = a[..., x_sorter]
    b = b[..., y_sorter]

    # 3 ----- Prepare and start FW

    # Initialize potentials
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    log = []

    pbar = trange(niter)
    for k in pbar:
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
        loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
        loss = loss + rho1 * torch.mean(kullback_leibler(A, a)) + rho2 * torch.mean(kullback_leibler(B, b))
        log.append(loss.data.item())

    # 4 ----- We are done. Get me out of here !
    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2)
    f, g = f + transl, g - transl
    A, B = a * torch.exp(-f / rho1), b * torch.exp(-g / rho2)
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    loss = loss + rho1 * torch.mean(kullback_leibler(A, a)) + rho2 * torch.mean(kullback_leibler(B, b))
    log.append(loss.data.item())

    # Reverse sort potentials and measures w.r.t order not sample (not sorted)
    f, g = torch.gather(f, 1, x_rev_sort), torch.gather(g, 1, y_rev_sort)
    A, B = torch.gather(A, 1, x_rev_sort), torch.gather(B, 1, y_rev_sort)
    
    return log, f, g, A, B, projections


if __name__ == '__main__':
    N, M, D = 400, 500, 10
    iter_fw = 5000
    num_proj = 15
    p=2
    rho1 = 1.


    a, x = generate_measure(N, D, slice=False)
    x = x + 0.5 * torch.ones_like(x)
    b, y = generate_measure(N, D, slice=False)

    log_usot, _, _, _, _, _ = USOT_store_loss(a, b, x, y, p, num_projections=num_proj, rho1=rho1, rho2=None, niter=iter_fw, mode='backprop', stochastic_proj=False)
    log_suot, _, _, _, _, _ = SUOT_store_loss(a, b, x, y, p, num_projections=num_proj, rho1=rho1, rho2=None, niter=iter_fw, mode='backprop')

    log_usot = np.asarray(log_usot)
    log_suot = np.asarray(log_suot)

    plt.figure()
    plt.plot( np.log((np.abs(log_suot[:1000] - log_suot[-2]))) , label='SUOT')
    plt.plot( np.log((np.abs(log_usot[:1000] - log_usot[-2]))) , label='USOT')
    plt.xlabel('number iter', fontsize=18)
    plt.ylabel('log(|loss - opt|)', fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig("./Convergence_FW_1000.pdf", format="pdf")
    plt.show()

    plt.figure()
    plt.plot( np.log((np.abs(log_suot[:200] - log_suot[-2]))) , label='SUOT')
    plt.plot( np.log((np.abs(log_usot[:200] - log_usot[-2]))) , label='USOT')
    plt.xlabel('number iter', fontsize=18)
    plt.ylabel('log(|loss - opt|)', fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig("./Convergence_FW_200.pdf", format="pdf")
    plt.show()

    plt.figure()
    plt.plot( np.log((np.abs(log_suot[:20] - log_suot[-2]))) , label='SUOT')
    plt.plot( np.log((np.abs(log_usot[:20] - log_usot[-2]))) , label='USOT')
    plt.xlabel('number iter', fontsize=18)
    plt.ylabel('log(|loss - opt|)', fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig("./Convergence_FW_20.pdf", format="pdf")
    plt.show()
    
