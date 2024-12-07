## Instructions to get the data

- Download the BBCSport from https://github.com/mkusner/wmd/tree/master (file `bbcsport-emd_tr_te_split.mat` from https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0)
- For the next datasets, first download `GoogleNews-vectors-negative300.bin.gz` from e.g. https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g, and unzip in the data folder.
- For MovieReviews, first unzip the file `data/review_polarity.zip`, and then launch
```
python dataset.py --dataset movie
```
The dataset comes from http://www.cs.cornell.edu/people/pabo/movie-review-data/ (`polarity dataset v2.0`).
- For the goodreads dataset, first download the dataset from https://ritual.uh.edu/multi_task_book_success_2017/ and then launch
```
python dataset.py --dataset goodreads
```


## Experiments

- Run `xp_doc_classif.py` with the right parameters to get the matrices of distance:
```
python xp_doc_classif.py --pbar --loss "usw" --n_projs 500 --dataset "BBC" --ntry 1 --rho1 0.0005 --rho2 0.0005 --inner_iter 10
```
- Get the results of kNN using the functions in `utils_knn.py`, see e.g. the notebook `kNN_BBC.ipynb`.

