# BBC
python xp_doc_classif.py --pbar --loss "w" --dataset "BBC" --ntry 1
python xp_doc_classif.py --pbar --loss "sw" --n_projs 500 --dataset "BBC" --ntry 3
python xp_doc_classif.py --pbar --loss "usw" --n_projs 500 --dataset "BBC" --ntry 3 --rho1 0.00021 --rho2 0.00021 --inner_iter 10
python xp_doc_classif.py --pbar --loss "stochastic_usw" --n_projs 500 --dataset "BBC" --ntry 3 --rho1 0.0002 --rho2 0.0002 --inner_iter 10
python xp_doc_classif.py --pbar --loss "suw" --n_projs 500 --dataset "BBC" --ntry 3 --rho1 0.01 --rho2 0.01 --inner_iter 10
python xp_doc_classif.py --pbar --loss "sinkhorn" --dataset "BBC" --ntry 1 --rho1 1 --rho2 1
python xp_doc_classif.py --pbar --loss "uw" --dataset "BBC" --ntry 1 --rho1 1 --rho2 1
