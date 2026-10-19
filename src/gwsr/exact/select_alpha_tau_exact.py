"""Association and objective functions used by the recorded current experiment."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gplearn.genetic import _parallel_evolve
from gplearn.fitness import _fitness_map
from gplearn.functions import _function_map
import original_bisquare as base
import paper_gp
K = 55
N_CHAINS = 24
INNER_SEED = 42
EPS = 1e-12

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    return dict(
        R2=float(r2_score(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        MAE=float(mean_absolute_error(y, p)),
    )


def haversine(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    lon_a = np.radians(coords_a[:, 0])[:, None]
    lat_a = np.radians(coords_a[:, 1])[:, None]
    lon_b = np.radians(coords_b[:, 0])[None, :]
    lat_b = np.radians(coords_b[:, 1])[None, :]
    dlon = lon_a - lon_b
    dlat = lat_a - lat_b
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def similarity_matrices(prepared: dict) -> tuple[np.ndarray, np.ndarray]:
    coords = prepared["query"][base.COORDS].to_numpy(dtype=float)
    dg = haversine(coords, coords)
    positive_d = dg[np.triu_indices_from(dg, k=1)]
    positive_d = positive_d[positive_d > 0]
    rho_g = float(np.median(positive_d)) if len(positive_d) else 1.0
    sg = np.exp(-dg / max(rho_g, EPS))
    df = cdist(prepared["descriptors"], prepared["descriptors"], metric="euclidean")
    positive_f = df[df > 0]
    rho_x = float(np.median(positive_f)) if len(positive_f) else 1.0
    sx = np.exp(-df / max(rho_x, EPS))
    np.fill_diagonal(sg, 1.0)
    np.fill_diagonal(sx, 1.0)
    prepared["rho_g_km"], prepared["rho_x"] = rho_g, rho_x
    return sg, sx


def projected(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(float(np.mean(coords[:, 1])))
    return np.column_stack([coords[:, 0] * 111.320 * np.cos(lat), coords[:, 1] * 110.574])


def build_order(sim: np.ndarray, coords: np.ndarray, tau: float) -> list[dict]:
    n = len(coords)
    nc = min(N_CHAINS, n)
    labels = KMeans(n_clusters=nc, random_state=INNER_SEED, n_init=20).fit_predict(projected(coords))
    rows: list[dict] = []
    for cid in range(nc):
        members = np.flatnonzero(labels == cid).tolist()
        if not members:
            continue
        if len(members) == 1:
            anchor = members[0]
        else:
            within = sim[np.ix_(members, members)].copy()
            np.fill_diagonal(within, 0.0)
            anchor = members[int(np.argmax(within.mean(axis=1)))]
        processed = [anchor]
        pending = set(members) - {anchor}
        rows.append(dict(chain_id=cid, order=0, query_id=anchor, parent=-1,
                         is_anchor=True, eligible=False, score=np.nan))
        order = 1
        while pending:
            bc = bp = None
            bs = -np.inf
            for child in sorted(pending):
                pp = processed[int(np.argmax(sim[child, processed]))]
                ss = float(sim[child, pp])
                if ss > bs:
                    bc, bp, bs = child, pp, ss
            assert bc is not None and bp is not None
            rows.append(dict(chain_id=cid, order=order, query_id=bc, parent=bp,
                             is_anchor=False, eligible=bool(bs >= tau), score=bs))
            processed.append(bc)
            pending.remove(bc)
            order += 1
    return rows


def objective(program, X, y, w, penalty):
    positive = w > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pred = program.execute(X)
        if not np.isfinite(pred[positive]).all():
            return np.inf
        loss = float(np.dot(w[positive], (y[positive] - pred[positive]) ** 2))
        value = loss + penalty * program.length_
    return float(value) if np.isfinite(value) else np.inf
