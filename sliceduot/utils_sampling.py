import torch
from .utils_hyperbolic import parallelTransport, expMap


def generate_measure(n_sample, n_dim, slice=True):
    """
    Generate a batch of probability measures in R^d sampled over
    the unit square
    :param n_batch: Number of batches
    :param n_sample: Number of sampling points in R^d
    :param n_dim: Dimension of the feature space
    :return: A (Nbatch, Nsample, Ndim) torch.Tensor
    """
    m = torch.distributions.exponential.Exponential(1.0)
    if slice:
        a = m.sample(torch.Size([n_sample, n_dim]))
        a = a / a.sum(dim=1)[:,None]
    else:
        a = m.sample(torch.Size([n_sample]))
        a = a / a.sum()
    m = torch.distributions.uniform.Uniform(0.0, 1.0)
    x = m.sample(torch.Size([n_sample, n_dim]))
    return a, x



def sampleWrappedNormal(mu, Sigma, n):
    device = mu.device
    
    d = len(mu)
    normal = torch.distributions.MultivariateNormal(torch.zeros((d-1,), device=device),Sigma)
    x0 = torch.zeros((1,d), device=device)
    x0[0,0] = 1
    
    ## Sample in T_x0 H
    v_ = normal.sample((n,))
    v = torch.nn.functional.pad(v_, (1,0))
    
    ## Transport to T_\mu H and project on H
    u = parallelTransport(v, x0, mu)    
    y = expMap(u, mu)
    
    return y 