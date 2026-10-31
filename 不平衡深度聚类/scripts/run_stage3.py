import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'results_stage3/mpl_cache'))
import argparse, ctypes, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
import igraph as ig
import leidenalg
from scipy.optimize import linear_sum_assignment
from scipy.special import ive
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'results_stage3/protocol_stage3.json'
MANIFEST = ROOT / 'results/feature_manifest.json'
spec = importlib.util.spec_from_file_location('stage2_evaluation', ROOT / 'scripts/run_stage2.py')
stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage2)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def arrsha(a):
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def peak_memory_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None

def subset(labels, k, ir, seed):
    rng = np.random.RandomState(seed)
    counts = np.array([int((50000 / k) * (1 / ir) ** (c / (k - 1))) for c in range(k)]) if ir != 1 else np.full(k, 50000 // k)
    return np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:counts[c]] for c in range(k)])

def load_data(dataset, ir, seed):
    # Reuse the stage-two constructor and its first-stage index-hash assertion.
    x, y, expected = stage2.load_data(dataset, ir, seed)
    return x, y, expected['indices_array_sha256']

def evaluate(y, pred, k, noise_cluster=None):
    pred = np.asarray(pred, dtype=np.int64)
    if noise_cluster is None:
        metrics = stage2.evaluate(y, pred, k)
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        stage1_tail = np.lexsort((np.arange(len(counts)), counts))[:len(counts) // 3]
        tab = np.zeros((k, len(counts)), dtype=np.int64)
        np.add.at(tab, (pred, y), 1)
        major = tab.argmax(axis=1)
        mtp = np.bincount(y[major[pred] == y], minlength=len(counts))
        metrics['many_to_one_stage1_tail_recall'] = float((mtp / counts)[stage1_tail].mean())
        metrics['stage1_tail_coverage'] = float(np.isin(stage1_tail, np.unique(major[tab.sum(1) > 0])).mean())
        q = np.bincount(pred, minlength=k) / len(pred)
        metrics['ARI'] = float(adjusted_rand_score(y, pred))
        metrics['predicted_clusters'] = int(np.unique(pred).size)
        metrics['normalized_entropy'] = float(-np.sum(q[q > 0] * np.log(q[q > 0])) / np.log(max(k, 2)))
        return metrics
    kt = int(y.max()) + 1
    tab = np.zeros((k, kt), dtype=np.int64)
    np.add.at(tab, (pred, y), 1)
    assignment_tab = tab.copy()
    if noise_cluster is not None: assignment_tab[noise_cluster] = 0
    rows, cols = linear_sum_assignment(-assignment_tab)
    mapping = np.full(k, -1, dtype=np.int64)
    mapping[rows] = cols
    if noise_cluster is not None: mapping[noise_cluster] = -1
    mapped = mapping[pred]
    counts = np.bincount(y, minlength=kt)
    tp = np.bincount(y[mapped == y], minlength=kt)
    pred_counts = np.bincount(mapped[mapped >= 0], minlength=kt)
    recall = tp / np.maximum(counts, 1)
    f1 = 2 * tp / np.maximum(counts + pred_counts, 1)
    ranked = np.lexsort((np.arange(kt), -counts))
    head_n = round(0.3 * kt)
    tail = ranked[kt - head_n:]
    medium = ranked[head_n:kt - head_n]
    stage1_tail = np.lexsort((np.arange(kt), counts))[:kt // 3]
    major = tab.argmax(axis=1)
    if noise_cluster is not None: major[noise_cluster] = -1
    mtp = np.bincount(y[major[pred] == y], minlength=kt)
    mrecall = mtp / np.maximum(counts, 1)
    covered = np.unique(major[(tab.sum(axis=1) > 0) & (major >= 0)])
    q = np.bincount(pred, minlength=k) / len(pred)
    positive = q > 0
    stage1_tail = np.lexsort((np.arange(kt), counts))[:kt // 3]
    return {'ACC': float((mapped == y).mean()), 'NMI': float(normalized_mutual_info_score(y, pred)),
            'ARI': float(adjusted_rand_score(y, pred)), 'balanced_accuracy': float(recall.mean()),
            'macro_F1': float(f1.mean()), 'head_recall': float(recall[ranked[:head_n]].mean()),
            'medium_recall': float(recall[medium].mean()), 'tail_recall': float(recall[tail].mean()),
            'stage1_tail_recall': float(recall[stage1_tail].mean()), 'many_to_one_BA': float(mrecall.mean()),
            'many_to_one_tail_recall': float(mrecall[tail].mean()), 'covered_classes': int(len(covered)),
            'tail_coverage': float(np.isin(tail, covered).mean()), 'many_to_one_stage1_tail_recall': float(mrecall[stage1_tail].mean()), 'stage1_tail_coverage': float(np.isin(stage1_tail, covered).mean()), 'predicted_clusters': int(np.unique(pred[pred != noise_cluster]).size) if noise_cluster is not None else int(np.unique(pred).size),
            'normalized_entropy': float(-np.sum(q[positive] * np.log(q[positive])) / np.log(max(k, 2)))}

def evaluate_majority(y, pred, k):
    pred = np.asarray(pred, dtype=np.int64)
    kt = int(y.max()) + 1
    tab = np.zeros((k, kt), dtype=np.int64)
    np.add.at(tab, (pred, y), 1)
    major = tab.argmax(axis=1)
    mapped = major[pred]
    counts = np.bincount(y, minlength=kt)
    tp = np.bincount(y[mapped == y], minlength=kt)
    rec = tp / np.maximum(counts, 1)
    pred_counts = np.bincount(mapped, minlength=kt)
    f1 = 2 * tp / np.maximum(counts + pred_counts, 1)
    stage1_tail = np.lexsort((np.arange(kt), counts))[:kt // 3]
    ranked = np.lexsort((np.arange(kt), -counts))
    h = round(0.3 * kt)
    tail = ranked[kt - h:]
    covered = np.unique(major[tab.sum(axis=1) > 0])
    q = np.bincount(pred, minlength=k) / len(pred)
    positive = q > 0
    return {'ACC': float((mapped == y).mean()), 'NMI': float(normalized_mutual_info_score(y, pred)),
            'ARI': float(adjusted_rand_score(y, pred)), 'balanced_accuracy': float(rec.mean()),
            'macro_F1': float(f1.mean()), 'head_recall': float(rec[ranked[:h]].mean()),
            'medium_recall': float(rec[ranked[h:kt-h]].mean()),
            'tail_recall': float(rec[tail].mean()), 'stage1_tail_recall': float(rec[stage1_tail].mean()), 'many_to_one_BA': float(rec.mean()),
            'many_to_one_tail_recall': float(rec[tail].mean()), 'covered_classes': int(len(covered)),
            'many_to_one_stage1_tail_recall': float(rec[stage1_tail].mean()),
            'tail_coverage': float(np.isin(tail, covered).mean()), 'stage1_tail_coverage': float(np.isin(stage1_tail, covered).mean()), 'predicted_clusters': int(np.unique(pred).size),
            'normalized_entropy': float(-np.sum(q[positive] * np.log(q[positive])) / np.log(k))}

def pca128(x, seed):
    return PCA(n_components=128, random_state=seed, svd_solver='randomized').fit_transform(x)

def finch_levels(x):
    nn = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=1).fit(x).kneighbors(return_distance=False)[:, 0]
    n = len(x)
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for i, j in enumerate(nn):
        union(i, int(j))
    first_by_neighbor = {}
    for i, j in enumerate(nn):
        j = int(j)
        if j in first_by_neighbor:
            union(i, first_by_neighbor[j])
        else:
            first_by_neighbor[j] = i
    labels = np.array([find(i) for i in range(n)])
    _, labels = np.unique(labels, return_inverse=True)
    levels = [labels.copy()]
    while len(np.unique(labels)) > 1 and len(levels) < 40:
        unique = np.unique(labels)
        centers = normalize(np.vstack([x[labels == z].mean(0) for z in unique]))
        next_nn = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=1).fit(centers).kneighbors(return_distance=False)[:, 0]
        parent = np.arange(len(unique))
        for i, j in enumerate(next_nn):
            a, b = i, int(j)
            while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
            while parent[b] != b: parent[b] = parent[parent[b]]; b = parent[b]
            if a != b: parent[b] = a
        first_by_neighbor = {}
        for i, j in enumerate(next_nn):
            j = int(j)
            if j in first_by_neighbor:
                a, b = i, first_by_neighbor[j]
                while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
                while parent[b] != b: parent[b] = parent[parent[b]]; b = parent[b]
                if a != b: parent[b] = a
            else:
                first_by_neighbor[j] = i
        for i in range(len(parent)):
            a = i
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            parent[i] = a
        labels = parent[labels]
        _, labels = np.unique(labels, return_inverse=True)
        levels.append(labels.copy())
    return levels

def finch_requested(x, levels, k):
    eligible = [z for z in levels if len(np.unique(z)) >= k]
    if not eligible:
        raise RuntimeError(f'FINCH first partition has fewer than requested {k} clusters')
    labels = eligible[-1].copy()
    while len(np.unique(labels)) > k:
        unique = np.unique(labels)
        centers = normalize(np.vstack([x[labels == z].mean(0) for z in unique]))
        distances = 1.0 - centers @ centers.T
        np.fill_diagonal(distances, np.inf)
        a, b = np.unravel_index(np.argmin(distances), distances.shape)
        labels[labels == unique[b]] = unique[a]
        _, labels = np.unique(labels, return_inverse=True)
    return labels

def over_merge(x, micro_k, target_k, seed):
    km = KMeans(micro_k, n_init=10, max_iter=300, random_state=seed).fit(x)
    labels = km.labels_.copy()
    centers = normalize(km.cluster_centers_)
    sizes = np.bincount(labels, minlength=micro_k).astype(float)
    dist = 1.0 - centers @ centers.T
    np.fill_diagonal(dist, np.inf)
    groups = np.arange(micro_k)
    active = np.ones(micro_k, dtype=bool)
    for _ in range(micro_k - target_k):
        a, b = np.unravel_index(np.argmin(dist), dist.shape)
        dist[a, :] = (sizes[a] * dist[a, :] + sizes[b] * dist[b, :]) / (sizes[a] + sizes[b])
        dist[:, a] = dist[a, :]
        sizes[a] += sizes[b]
        groups[groups == b] = a
        active[b] = False
        dist[b, :] = np.inf; dist[:, b] = np.inf; dist[a, a] = np.inf
    _, inverse = np.unique(groups, return_inverse=True)
    merged = inverse[labels]
    return merged, labels

def fit_method(x, y, k, seed, method, kprime=None):
    fit = {}
    if method == 'GMM_diag':
        z = pca128(x, seed)
        model = GaussianMixture(n_components=k, covariance_type='diag', n_init=3, reg_covar=1e-4, init_params='kmeans', random_state=seed)
        pred = model.fit_predict(z)
        fit = {'lower_bound': float(model.lower_bound_)}
    elif method == 'SC_knn':
        knn = NearestNeighbors(n_neighbors=11, metric='cosine', n_jobs=1).fit(x)
        distances, indices = knn.kneighbors(x)
        affinity = np.zeros((len(x), len(x)), dtype=np.float32)
        rows = np.repeat(np.arange(len(x)), 10)
        affinity[rows, indices[:, 1:].reshape(-1)] = 1.0 - distances[:, 1:].reshape(-1)
        affinity = np.maximum(affinity, affinity.T)
        np.fill_diagonal(affinity, 1.0)
        pred = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='cluster_qr', random_state=seed).fit_predict(affinity)
    elif method == 'Leiden_knn':
        knn = NearestNeighbors(n_neighbors=11, metric='cosine', n_jobs=1).fit(x)
        distances, indices = knn.kneighbors(x)
        edges = {}
        for i in range(len(x)):
            for j, d in zip(indices[i, 1:], distances[i, 1:]):
                a, b = sorted((i, int(j)))
                edges[(a, b)] = max(edges.get((a, b), 0.0), 1.0 - float(d))
        graph = ig.Graph(n=len(x), edges=list(edges), directed=False)
        graph.es['weight'] = list(edges.values())
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                             weights='weight', resolution_parameter=1.0, seed=seed)
        pred = np.asarray(partition.membership, dtype=np.int64)
    elif method == 'AGG_ward':
        pred = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(pca128(x, seed))
    elif method == 'AGG_avg_cos':
        pred = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine').fit_predict(x)
    elif method == 'OVER_MERGE':
        pred, micro = over_merge(x, 3 * k, k, seed)
        fit = {'microcluster_counts': np.bincount(micro, minlength=3 * k).tolist(), 'premerge_metrics': evaluate(y, micro, 3 * k)}
    elif method == 'OVER_ORACLE_MERGE':
        model = KMeans(kprime, n_init=10, max_iter=300, random_state=seed).fit(x)
        pred = model.labels_
        fit = {'microclusters': int(kprime), 'inertia': float(model.inertia_)}
    elif method.startswith('FINCH'):
        levels = finch_levels(x)
        if method == 'FINCH_reqK':
            pred = finch_requested(x, levels, k)
        else:
            sample = np.random.RandomState(seed).permutation(len(x))[:2000]
            def score(z):
                count = len(np.unique(z[sample]))
                return silhouette_score(x[sample], z[sample], metric='euclidean') if 1 < count < len(sample) else -1
            pred = max(levels, key=score)
        fit = {'levels': [int(len(np.unique(z))) for z in levels]}
    elif method in ('HDBSCAN_error', 'HDBSCAN_nearest'):
        model = HDBSCAN(min_cluster_size=5, min_samples=5, metric='euclidean', n_jobs=1)
        z = pca128(x, seed)
        raw = model.fit_predict(z).copy()
        raw_noise = int(np.sum(raw < 0))
        raw_clusters = int(len(np.unique(raw[raw >= 0])))
        if np.any(raw < 0):
            if method == 'HDBSCAN_error':
                noise_cluster = int(raw.max()) + 1
                pred = np.where(raw < 0, noise_cluster, raw)
            else:
                known = np.unique(raw[raw >= 0])
                centers = np.vstack([z[raw == c].mean(0) for c in known])
                raw[raw < 0] = known[((z[raw < 0, None] - centers[None]) ** 2).sum(2).argmin(1)]
                pred = raw
        else:
            pred = raw
        pred = np.asarray(pred, dtype=np.int64)
        fit = {'raw_noise': raw_noise, 'raw_clusters': raw_clusters}
        if method == 'HDBSCAN_error' and fit['raw_noise']: fit['noise_cluster'] = noise_cluster
    elif method == 'vMF_free':
        km = KMeans(k, n_init=10, max_iter=300, random_state=seed).fit(x)
        mean_norms = np.linalg.norm(km.cluster_centers_, axis=1)
        means = normalize(km.cluster_centers_)
        weights = np.bincount(km.labels_, minlength=k).astype(float) / len(x)
        dimension = x.shape[1]
        def concentration(r):
            r = np.clip(np.asarray(r, dtype=np.float64), 1e-4, 0.9999)
            return np.clip(r * (dimension - r * r) / (1 - r * r), 100.0, 10000.0)
        kappa = concentration(mean_norms)
        previous = None
        for iteration in range(100):
            nu = dimension / 2 - 1
            log_normalizer = nu * np.log(kappa) - (dimension / 2) * np.log(2 * np.pi) - np.log(np.maximum(ive(nu, kappa), 1e-300)) - kappa
            logp = x @ means.T * kappa + np.log(np.maximum(weights, 1e-12)) + log_normalizer
            logp -= logp.max(axis=1, keepdims=True)
            resp = np.exp(logp); resp /= resp.sum(axis=1, keepdims=True)
            nk = resp.sum(axis=0)
            weights = nk / len(x)
            resultant = resp.T @ x
            means = normalize(resultant)
            kappa = concentration(np.linalg.norm(resultant, axis=1) / np.maximum(nk, 1e-12))
            pred = resp.argmax(axis=1)
            if previous is not None and np.array_equal(previous, pred):
                break
            previous = pred.copy()
        fit = {'iterations': iteration + 1}
    else:
        raise ValueError(method)
    return pred.astype(np.int64), fit

def run(args):
    assert sha(PROTOCOL) == PROTOCOL.with_suffix('.sha256').read_text().split()[0].lower()
    stage2.setup(args.seed)
    x, y, indices_hash = load_data(args.dataset, args.ir, args.seed)
    k = 10 if args.dataset == 'cifar10' else 100
    start = time.perf_counter()
    pred, fit = fit_method(x, y, k, args.seed, args.method, args.kprime)
    eval_k = args.kprime if args.method == 'OVER_ORACLE_MERGE' else (int(pred.max()) + 1 if args.method in ('HDBSCAN_error', 'HDBSCAN_nearest', 'FINCH_auto', 'Leiden_knn') else k)
    metrics = evaluate_majority(y, pred, eval_k) if args.method == 'OVER_ORACLE_MERGE' else evaluate(y, pred, eval_k, fit.get('noise_cluster'))
    out_k = args.kprime if args.kprime else k
    out = ROOT / args.out / f'{args.dataset}_ir{args.ir}_s{args.seed}' / f'{args.method}_k{out_k}' / 'result.json'
    dump(out, {'dataset': args.dataset, 'ir': args.ir, 'seed': args.seed, 'method': args.method, 'k': k, 'n': len(y), 'protocol_sha256': sha(PROTOCOL), 'indices_array_sha256': indices_hash, 'prediction_sha256': arrsha(pred), 'code_sha256': sha(Path(__file__)), 'metrics': metrics, 'fit': fit, 'elapsed_s': time.perf_counter() - start, 'peak_memory_bytes': peak_memory_bytes()})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--ir', type=int, choices=[1, 10, 100], required=True)
    parser.add_argument('--seed', type=int, choices=[1, 2, 3, 4, 5], required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--kprime', type=int)
    parser.add_argument('--out', default='results_stage3/full')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)
