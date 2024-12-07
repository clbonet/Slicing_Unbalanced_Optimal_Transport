import torch
import pickle
import os
import numpy as np
np.random.seed(10)

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'
# plt.rc('text', usetex=True)

from sliceduot.sliced_uot import unbalanced_sliced_ot, sliced_unbalanced_ot

cmap = matplotlib.cm.get_cmap('Set2') 


def sample_complexity_usot(list_dims, list_samples, rho_1, rho_2, n_runs):
    # Compute USOT for different parameters
    usot = np.zeros((n_runs, list_dims.shape[0], list_samples.shape[0]))
    suot = np.zeros((n_runs, list_dims.shape[0], list_samples.shape[0]))
    for nr in range(n_runs):
        print("Run {}...".format(nr+1))
        for d in range(list_dims.shape[0]):
            dim = list_dims[d]
            print("\t\t Dimension: {}".format(dim))
            for n in range(list_samples.shape[0]):
                n_samples = list_samples[n]
                print("\t\t\t Number of samples: {}".format(n_samples))
                # Sample first dataset
                x1 = torch.normal(0, 1, size=(n_samples, dim))
                w1 = torch.ones(n_samples) / n_samples
                # Sample second dataset
                x2 = torch.normal(0, 1, size=(n_samples, dim))
                w2 = torch.ones(n_samples) / n_samples
                # Compute USOT between x1 and x2
                loss, _, _, _, _, _ = unbalanced_sliced_ot(w1, w2, x1, x2, p=2, num_projections=100, rho1=rho_1, rho2=rho_2, niter=200, mode="backprop")
                usot[nr, d, n] = loss
                # Store USOT results
                with open("usot_dims={}_nruns={}_rho1={}_rho2={}".format([i for i in list_dims], n_runs, rho_1, rho_2), "wb") as f:
                    pickle.dump(usot, f, pickle.HIGHEST_PROTOCOL)
                
                # Compute SUOT between x1 and x2
                loss_2, _, _, _, _, _ = sliced_unbalanced_ot(w1, w2, x1, x2, p=2, num_projections=100, rho1=rho_1, rho2=rho_2, niter=200, mode="backprop")  
                suot[nr, d, n] = loss_2
                # Store USOT results
                with open("suot_dims={}_nruns={}_rho1={}_rho2={}".format([i for i in list_dims], n_runs, rho_1, rho_2), "wb") as f:
                    pickle.dump(suot, f, pickle.HIGHEST_PROTOCOL)


    # # Load USOT/SUOT results
    # with open("suot_dims={}_nruns={}_rho1={}_rho2={}".format([i for i in list_dims], n_runs, rho_1, rho_2), "rb") as f:
    #    suot = pickle.load(f)
    # with open("usot_dims={}_nruns={}_rho1={}_rho2={}".format([i for i in list_dims], n_runs, rho_1, rho_2), "rb") as f:
    #    usot = pickle.load(f)

    fig = plt.figure()
    for d in range(list_dims.shape[0]):
        dim = list_dims[d]
        usot_mean = usot[:, d, :].mean(axis=0)
        usot_10 = np.percentile(usot[:, d, :], 10, axis=0)
        usot_90 = np.percentile(usot[:, d, :], 90, axis=0)
        plt.loglog(list_samples, usot_mean, label=r"USOT, d = {}".format(dim), color=cmap(d), lw=1.5)
        plt.fill_between(list_samples, usot_10, usot_90, facecolor=cmap(d), alpha=0.2)
        suot_mean = suot[:, d, :].mean(axis=0)
        suot_10 = np.percentile(suot[:, d, :], 10, axis=0)
        suot_90 = np.percentile(suot[:, d, :], 90, axis=0)
        plt.loglog(list_samples, suot_mean, ls='--', label=r"SUOT, d = {}".format(dim), color=cmap(d), lw=1.5)
        plt.fill_between(list_samples, suot_10, suot_90, facecolor=cmap(d), alpha=0.2)
    
    plt.loglog(list_samples, 1/list_samples, label=r"$1/n$", color='k', lw=1.5)
    plt.legend()
    plt.title(r'$\rho_1 = {}, \rho_2 = {}$'.format(rho_1, rho_2))
    plt.xlabel(r"number of samples $n$")
    plt.ylabel("loss")
    fig.savefig("complexity_nruns={}_rho1={}_rho2={}.pdf".format(n_runs, rho_1, rho_2), bbox_inches='tight')




if __name__ == "__main__":
    # Set parameters
    # list_proj = np.array([1, 10, 100, 1000, 10000])  # number of projections in the Monte Carlo approximation
    list_dims = np.array([5, 10, 20, 50, 100])  # list of dimension values
    list_samples = np.array([50, 100, 500, 1000, 5000])  # different number of samples for the generated datasets
    n_runs = 30

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rho_1, rho_2 = 0.1, 0.1

    # Compute divergences and plot figures
    sample_complexity_usot(list_dims, list_samples, rho_1, rho_2, n_runs)
