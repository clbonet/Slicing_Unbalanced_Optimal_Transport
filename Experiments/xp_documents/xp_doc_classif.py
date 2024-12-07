import torch
import sys
import argparse
import ot
import time

import scipy.io as sio
import numpy as np
import pandas as pd

from tqdm.auto import trange
from joblib import Parallel, delayed

from sliceduot.sliced_uot import unbalanced_sliced_ot, sliced_unbalanced_ot

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="sw", help="Which loss to use")
parser.add_argument("--dataset", type=str, default="Twitter", help="Which dataset to use")
parser.add_argument("--n_projs", type=int, default=500, help="Number of projections")
parser.add_argument("--inner_iter", type=int, default=50, help="Number of inner iter of suot or rsot")
parser.add_argument("--rho1", type=float, default=1, help="rho1")
parser.add_argument("--rho2", type=float, default=1, help="rho2")
parser.add_argument("--reg_sinkhorn", type=float, default=0.1, help="Epsilon sinkhorn")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
# parser.add_argument("--draw_once", action="store_true", help="If yes, draw once the projections for RSW")
parser.add_argument("--unnormalize", action="store_true", help="If yes, does not normalize measures")
parser.add_argument("--njobs", type=int, default=5, help="Number of jobs in parallel")
parser.add_argument("--ntry", type=int, default=5, help="Number of try")
parser.add_argument("--size_batch", type=int, default=50, help="Size of batchs")
parser.add_argument("--num_batch", type=int, default=10, help="Number of batchs")
args = parser.parse_args()


def compute_ot(i):
#     print("i launched", i, device, flush=True)
    L = range(i+1, len(X_train))

    for j in L:
        x1 = torch.tensor(X_train[i], device=device, dtype=torch.float64).T
        w1 = torch.tensor(w_train[i], device=device, dtype=torch.float64)[0]

        x2 = torch.tensor(X_train[j], device=device, dtype=torch.float64).T
        w2 = torch.tensor(w_train[j], device=device, dtype=torch.float64)[0]
                
        if not args.unnormalize:
            w1 /= np.sum(w_train[i])
            w2 /= np.sum(w_train[j])

        if args.loss == "sw":
            t = time.time()
            loss = ot.sliced_wasserstein_distance(x1, x2, w1, w2, n_projs)**2
            ts.append(time.time()-t)
            
        elif args.loss == "stochastic_usw":
            t = time.time()
            loss, _, _, _, _, _ = unbalanced_sliced_ot(w1, w2, x1, x2, p=2, num_projections=n_projs, 
                                                       rho1=args.rho1, rho2=args.rho2, niter=args.inner_iter,
                                                       stochastic_proj=True, mode="backprop")  
            ts.append(time.time()-t)
            
        elif args.loss == "usw":
            t = time.time()
            loss, _, _, _, _, _ = unbalanced_sliced_ot(w1, w2, x1, x2, p=2, num_projections=n_projs, 
                                                       rho1=args.rho1, rho2=args.rho2, niter=args.inner_iter,
                                                       mode="backprop")  
            ts.append(time.time()-t)

        elif args.loss == "suw":
            t = time.time()
            loss, _, _, _, _, _ = sliced_unbalanced_ot(w1, w2, x1, x2, p=2, num_projections=n_projs, 
                                                       rho1=args.rho1, rho2=args.rho2, niter=args.inner_iter,
                                                       mode="backprop")  
            ts.append(time.time()-t)

        elif args.loss == "uw":
            t = time.time()
            M = ot.dist(x1, x2, metric="sqeuclidean")
            loss = ot.unbalanced.mm_unbalanced2(w1, w2, M, reg_m=args.rho1)
            ts.append(time.time())
            
        elif args.loss == "sinkhorn":
            t = time.time()
            M = ot.dist(x1, x2, metric="sqeuclidean")
            M /= M.max()
            loss = ot.unbalanced.sinkhorn_unbalanced2(w1, w2, M, reg=args.reg_sinkhorn, reg_m=args.rho1, method="sinkhorn_stabilized")
            ts.append(time.time()-t)

        elif args.loss == "w":
            t = time.time()
            M = ot.dist(x1, x2, metric="sqeuclidean")
            loss = ot.emd2(w1, w2, M)
            ts.append(time.time()-t)


        dist_mat[i, j] = loss.item()
        dist_mat[j, i] = loss.item()
    

if __name__ == "__main__":
    
    print(device, args.loss, flush=True)
    
    n_projs = args.n_projs
        
    if args.dataset == "BBC":
        mat_contents = sio.loadmat("./data/bbcsport-emd_tr_te_split.mat")

        X = mat_contents["X"][0]
        w = mat_contents["BOW_X"][0]

    elif args.dataset == "movie":
#         X, w, _ = get_movie_review()
        X = np.load("./data/X_movie.npy", allow_pickle=True)
        w = np.load("./data/w_movie.npy", allow_pickle=True)
        
    elif args.dataset == "goodreads":
        X = np.load("./data/X_goodread.npy", allow_pickle=True)
        w = np.load("./data/w_goodread.npy", allow_pickle=True)
        
    n_try = args.ntry
        
    
    for k in range(n_try):
        X_train = X
        w_train = w
       
        if args.pbar:
            pbar = trange(len(X_train))
        else:
            pbar = range(len(X_train))
            
        dist_mat = np.zeros((len(X_train), len(X_train)))
        ts = []
                                

#         for i in pbar:
        Parallel(n_jobs=args.njobs, require="sharedmem")(delayed(compute_ot)(i) for i in pbar)
        
        if (args.loss == "usw" or args.loss == "stochastic_usw" or args.loss == "suw") and args.unnormalize:
            np.savetxt("./results_"+str(args.dataset)+"/d_projs"+str(n_projs)+"_"+args.loss+"_unnormalize_"+ \
                       args.dataset+"_rho1"+str(args.rho1)+"_rho2"+str(args.rho2)+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_projs"+str(n_projs)+"_"+args.loss+"_unnormalize_"+args.dataset+"_rho1"+ \
                       str(args.rho1)+"_rho2"+str(args.rho2)+"_k"+str(k), ts)
        
        elif args.loss == "usw" or args.loss == "stochastic_usw" or args.loss == "suw":
            np.savetxt("./results_"+str(args.dataset)+"/d_projs"+str(n_projs)+"_"+args.loss+"_"+ \
                       args.dataset+"_rho1"+str(args.rho1)+"_rho2"+str(args.rho2)+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_projs"+str(n_projs)+"_"+args.loss+"_"+args.dataset+"_rho1"+ \
                       str(args.rho1)+"_rho2"+str(args.rho2)+"_k"+str(k), ts)
            
        elif args.loss == "sw":
            np.savetxt("./results_"+str(args.dataset)+"/d_projs"+str(n_projs)+"_"+args.loss+"_"+args.dataset+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_projs"+str(n_projs)+"_"+args.loss+"_"+args.dataset+"_k"+str(k), ts)

            
        elif args.loss == "sinkhorn":
            np.savetxt("./results_"+str(args.dataset)+"/d_"+args.loss+"_"+args.dataset+"_rho"+ \
                       str(args.rho1)+"_reg"+str(args.reg_sinkhorn)+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_"+args.loss+"_"+args.dataset+"_rho"+ \
                       str(args.rho1)+"_reg"+str(args.reg_sinkhorn)+"_k"+str(k), ts)
            
        elif args.loss == "uw":
            np.savetxt("./results_"+str(args.dataset)+"/d_"+args.loss+"_"+args.dataset+"_rho"+ \
                       str(args.rho1)+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_"+args.loss+"_"+args.dataset+"_rho"+ \
                       str(args.rho1)+"_k"+str(k), ts)
            
        elif args.loss == "muw":
            np.savetxt("./results_"+str(args.dataset)+"/d_"+args.loss+"_"+args.dataset+"_rho"+ \
                       str(args.rho1)+"_nbatch"+str(args.num_batch)+"_sizebatch"+ \
                       str(args.size_batch)+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_"+args.loss+"_"+args.dataset+"_rho"+ \
                       str(args.rho1)+"_nbatch"+str(args.num_batch)+"_sizebatch"+ \
                       str(args.size_batch)+"_k"+str(k), ts)
            
        else:
            np.savetxt("./results_"+str(args.dataset)+"/d_"+args.loss+"_"+args.dataset+"_k"+str(k), dist_mat)
            np.savetxt("./results_time/ts_"+args.loss+"_"+args.dataset+"_k"+str(k), ts)
