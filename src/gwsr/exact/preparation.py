"""Fold-fitted transformations and eight-dimensional target-point descriptors."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import original_bisquare as base
import select_alpha_tau_exact as exp

def prepare_training(raw_train, query):
    """No query responses used for cleaning, transformations or distances."""
    q1, q3, iqr, lower, upper = base.iqr_bounds(raw_train[base.TARGET], 3.0)
    keep = raw_train[base.TARGET].between(lower, upper)
    train = raw_train.loc[keep].copy()
    xs, ys = StandardScaler(), StandardScaler()
    xtrain = xs.fit_transform(train[base.FEATURES].to_numpy())
    xquery = xs.transform(query[base.FEATURES].to_numpy())
    ytrain = ys.fit_transform(np.log(train[base.TARGET].to_numpy()).reshape(-1, 1)).ravel()
    distances = exp.haversine(query[base.COORDS].to_numpy(), train[base.COORDS].to_numpy())
    info = dict(raw_train_count=len(raw_train), retained_train_count=len(train),
                removed_train_count=int((~keep).sum()), lower=lower, upper=upper,
                q1=q1, q3=q3, iqr=iqr,
                retained_row_ids=train['row_id'].astype(int).tolist(),
                removed_row_ids=raw_train.loc[~keep, 'row_id'].astype(int).tolist(),
                x_mean=xs.mean_.tolist(), x_scale=xs.scale_.tolist(),
                log_y_mean=ys.mean_.tolist(), log_y_scale=ys.scale_.tolist())
    return xtrain, ytrain, xquery, distances, ys, info


def prepare_fold(raw_train, query, k=55):
    K = k
    assert exp.base.TARGET not in query
    xfit, yfit, xquery, _, ys, info = prepare_training(raw_train, query)
    clean = raw_train.set_index("row_id").loc[info["retained_row_ids"]].reset_index()
    distances = exp.haversine(query[exp.base.COORDS].to_numpy(), clean[exp.base.COORDS].to_numpy())
    positions = np.argsort(distances, axis=1)[:, :K]
    weights = []
    for d in distances:
        _, _, w = exp.base.get_local_sample(-1, d, xfit, yfit, K)
        weights.append(w)
    return dict(xfit=xfit, yfit=yfit, xquery=xquery, distances=distances,
                positions=positions, weights=np.asarray(weights), descriptors=xquery.copy(),
                ys=ys, query=query.reset_index(drop=True).copy(), train=clean, preprocessing=info)
