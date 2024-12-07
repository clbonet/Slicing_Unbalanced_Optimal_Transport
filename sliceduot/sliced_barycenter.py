import torch
from tqdm import tqdm
from .sliced_uot import unbalanced_sliced_ot, sliced_ot


def compute_barycenter_grid_MD_unbalanced(Xgrid, full_data, weights = None, rho1=1, rho2=None, bar_iter=100, 
                                           fw_iter = 10, lr_bar=1, num_proj=16, p=2, mode='backprop'):
    """Computes the barycenter measure over a grid w.r.t to the distance USOT or balanced SOT.

    Args:
        Xgrid (torch.tensor of size [N^2,D]): Support of the barycenter in R^D, typically a regular grid.
        full_data (torch.tensor of size [K,N^2]): weights of K input measures of support Xgrid
        weights (torch.Tensor of size [K]): weights of barycentric loss, i.e. importance of input measures. 
                                            Defaults to None for uniform weights (isobarycenter).
        rho1 (int, optional): relaxation parameter w.r.t input measures. Defaults to 1. None to obtain Balanced SOT barycenter.
        rho2 (_type_, optional): relaxation parameter w.r.t barycenter. Defaults to None when set equal to rho1.
        bar_iter (int, optional): Number of Nesterov Mirror GD to compute barycenter. Defaults to 100.
        fw_iter (int, optional): Number of FW iterations to compute RSOT for unbalanced barycenter. Defaults to 10.
        lr_bar (int, optional): Learning rate of Nesterov Mirror GD to compute barycenter. Defaults to 1.
        num_proj (int, optional): Number of projections in SOT/RSOT. Defaults to 16.
        p (int, optional): Exponents of the cost |x-y|^p in SOT/RSOT loss. Defaults to 2.
        mode (str, optional): Method to estimate gradient w.r.t. barycenter weights. Defaults to 'backprop', for estimation via backprop. 
                                Take 'explicit' to estimate gradients via closed form.

    Returns:
        barycenter (torch.tensor of size [N^2]): weights of the barycenter supported over Xgrid.
    """

    # Init default values
    num_T = full_data.shape[0]
    assert mode in ['backprop', 'explicit']
    if weights is None:
        weights = torch.ones(num_T).to(full_data.device) / num_T
    if (rho2 is None) or (rho1 is None):
        rho2 = rho1

    # init barycenter
    barycenter = torch.ones_like(full_data[0]) / full_data.shape[1]
    #print('test -*---')
    #barycenter = full_data.mean(0)
    if mode == 'backprop':
        barycenter.requires_grad_(True)
    if mode == 'explicit':
        g = torch.zeros_like(barycenter)

    pbar = tqdm(range(bar_iter))
    
    tab_loss= []
    barhat = barycenter.clone().detach()
    bartilde = barycenter.clone().detach()

    for i in pbar:
        # Nesterov Acceleration
        beta = (i+1)/2
        barycenter.data = (1-1./beta) * barhat + bartilde / beta
        
        loss = torch.Tensor([0.]).to(full_data.device)
        for t in range(num_T):
            if rho1 is None: # Balanced SOT
                val, _, gb, _ = sliced_ot(full_data[t], barycenter, Xgrid, Xgrid,  p=p, num_projections=num_proj)
            else: # Unblanced
                val, _, gb, _, _, _ = unbalanced_sliced_ot(full_data[t], barycenter, Xgrid, Xgrid,  p=p, num_projections=num_proj, 
                                                          rho1=rho1, rho2=rho2, niter=fw_iter)
            loss += weights[t] * val
            if mode == 'explicit': # Estimate gradient via closed formula
                if rho1 is None:
                    g += gb * weights[t]
                else:
                    g += rho2 *(1 - (-gb / rho2).exp()) * weights[t]

        # center grad
        if mode == 'backprop':
            loss.backward()
            grad_u = barycenter.grad - barycenter.grad.mean()
            barycenter.grad.zero_()
        if mode == 'explicit':
            grad_u = g - g.mean()
            g.zero_()

        #Prox update
        bartilde = bartilde * torch.exp(- lr_bar * beta * grad_u) # multiplicative update
        bartilde = bartilde / bartilde.sum() # rescale as probability

        # final Nesterov update
        barhat = (1-1./beta) * barhat + bartilde / beta

        
        tab_loss.append(loss.item())
        pbar.set_description("Objective value {}".format(tab_loss[-1]))

    return barycenter