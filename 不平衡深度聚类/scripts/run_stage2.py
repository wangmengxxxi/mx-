"""Stage-two feature diagnostics. Third-party solver is imported from audited local source."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results_stage2/protocol_stage2.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def setup(seed):
    assert sha(PROTOCOL) == (ROOT / "results_stage2/protocol_stage2.sha256").read_text().split()[0]
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_capability(0) == (12, 0)


def load_data(dataset, ir, seed):
    ktrue = 10 if dataset == "cifar10" else 100
    manifest = json.loads((ROOT / "results/feature_manifest.json").read_text(encoding="utf-8"))
    fpath = ROOT / f"data/features/{dataset}_train.npy"
    lpath = ROOT / f"data/features/{dataset}_train_labels.npy"
    assert sha(fpath) == manifest[f"{dataset}_train"]["sha256"]
    assert sha(lpath) == manifest[f"{dataset}_train"]["labels_sha256"]
    labels = np.load(lpath)
    counts = np.array([int((50000 / ktrue) * (1 / ir) ** (c / (ktrue - 1))) for c in range(ktrue)])
    rng = np.random.RandomState(seed)
    ids = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:counts[c]] for c in range(ktrue)])
    expected = json.loads((ROOT / f"results/full/{dataset}_ir{ir}_s{seed}/result.json").read_text(encoding="utf-8"))
    assert array_sha(ids) == expected["indices_array_sha256"]
    x = normalize(np.load(fpath, mmap_mode="r")[ids].astype(np.float32), norm="l2")
    return np.ascontiguousarray(x), labels[ids], expected


def evaluate(y, pred, k):
    ktrue = int(y.max()) + 1
    tab = np.zeros((k, ktrue), dtype=np.int64)
    np.add.at(tab, (pred, y), 1)
    rows, cols = linear_sum_assignment(-tab)
    mapping = np.full(k, -1, dtype=np.int64)
    mapping[rows] = cols
    mapped = mapping[pred]
    true_counts = np.bincount(y, minlength=ktrue)
    tp = np.bincount(y[(mapped == y)], minlength=ktrue)
    pred_counts = np.bincount(mapped[mapped >= 0], minlength=ktrue)
    recall = tp / true_counts
    f1 = 2 * tp / np.maximum(true_counts + pred_counts, 1)
    ranked = np.lexsort((np.arange(ktrue), -true_counts))
    h = int(round(ktrue * 0.3))
    t = int(round(ktrue * 0.3))
    head, medium, tail = ranked[:h], ranked[h:ktrue-t], ranked[ktrue-t:]
    stage1_tail = np.lexsort((np.arange(ktrue), true_counts))[:ktrue // 3]
    major = tab.argmax(axis=1)
    mtp = np.bincount(y[(major[pred] == y)], minlength=ktrue)
    mrecall = mtp / true_counts
    covered = np.unique(major[tab.sum(axis=1) > 0])
    return {
        "ACC": float((mapped == y).mean()),
        "NMI": float(normalized_mutual_info_score(y, pred, average_method="arithmetic")),
        "balanced_accuracy": float(recall.mean()), "macro_F1": float(f1.mean()),
        "head_recall": float(recall[head].mean()), "medium_recall": float(recall[medium].mean()),
        "tail_recall": float(recall[tail].mean()),
        "stage1_tail_recall": float(recall[stage1_tail].mean()),
        "many_to_one_BA": float(mrecall.mean()),
        "many_to_one_tail_recall": float(mrecall[tail].mean()),
        "covered_classes": int(len(covered)),
        "tail_coverage": float(np.isin(tail, covered).mean()),
        "recall_per_class": recall.tolist(), "mapping": mapping.tolist(),
        "cluster_counts": np.bincount(pred, minlength=k).tolist(),
    }


class CosHead(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.embedding = nn.utils.weight_norm(nn.Linear(dim, k, bias=False))
        self.embedding.weight_g.data.fill_(1)
        self.embedding.weight_g.requires_grad = False

    def forward(self, x):
        return 10 * self.embedding(F.normalize(x, dim=1))


def official_solver():
    file = next((ROOT / "third_party").glob("PPOT-*/utils/sinkhorn_knopp.py"))
    spec = importlib.util.spec_from_file_location("reviewed_ppot_solver", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SemiCurrSinkhornKnopp(epsilon=0.1, gamma=1, stoperr=1e-6,
                                         numItermax=1000, rho=0.1, semi_use=True)


class PriorSolver:
    """Paper Algorithm 1 with class-size marginals; implementation local to diagnosis."""
    def __init__(self, prior):
        self.prior = torch.as_tensor(prior, dtype=torch.float64, device="cuda")
        self.rho = 0.1
        self.b = None

    @torch.no_grad()
    def __call__(self, logits):
        p = -torch.log(torch.softmax(logits.detach().double(), dim=1))
        n, k = p.shape
        cost = torch.cat((p, torch.zeros(n, 1, dtype=torch.float64, device="cuda")), dim=1)
        q = torch.exp(-cost / 0.1)
        pa = torch.full((n, 1), 1 / n, dtype=torch.float64, device="cuda")
        pb = torch.cat((self.rho * self.prior[:, None],
                        torch.tensor([[1 - self.rho]], dtype=torch.float64, device="cuda")))
        b = torch.ones(k + 1, 1, dtype=torch.float64, device="cuda") / (k + 1) if self.b is None else self.b
        fi = 1 / 1.1
        for _ in range(1000):
            a = pa / (q @ b)
            new_b = pb / (q.T @ a)
            new_b[:-1] = new_b[:-1].pow(fi)
            error = torch.norm(new_b - b)
            b = new_b
            if error <= 1e-6:
                break
        self.b = b
        return (n * a * q * b.T)[:, :-1].float()


def fit_ppot(x, k, seed, epochs, prior_sizes=None):
    torch.manual_seed(seed)
    heads = nn.ModuleList([CosHead(x.shape[1], k), CosHead(x.shape[1], k)]).cuda()
    xt = torch.from_numpy(x).cuda()
    optim = torch.optim.Adam((p for p in heads.parameters() if p.requires_grad), lr=5e-4)
    gen = torch.Generator().manual_seed(seed)
    solvers = [official_solver() for _ in range(2)]
    bank = None
    prior_rank = None
    best_loss = float("inf")
    best_state = None
    best_head = None
    history = []
    total_steps = epochs * math.ceil(len(x) / 512)
    step = 0
    bank_ids = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:5120].cuda()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        if epoch == 1:
            with torch.no_grad():
                bank = [h(xt[bank_ids]).detach().clone() for h in heads]
                if prior_sizes is not None:
                    prior_rank = [np.argsort(-v.argmax(1).bincount(minlength=k).cpu().numpy(), kind="stable") for v in bank]
                    sizes = np.sort(np.asarray(prior_sizes, dtype=np.float64))[::-1]
                    old_solvers = solvers
                    solvers = []
                    for rank in prior_rank:
                        prior = np.empty(k, dtype=np.float64)
                        prior[rank] = sizes / sizes.sum()
                        solvers.append(PriorSolver(prior))
                    for j in range(2):
                        solvers[j].b = old_solvers[j].b
        lr = 5e-7 + (5e-4 - 5e-7) * (1 + math.cos(math.pi * epoch / epochs)) / 2
        for group in optim.param_groups:
            group["lr"] = lr
        perm = torch.randperm(len(x), generator=gen).cuda()
        losses = np.zeros(2)
        for indices in perm.split(512):
            batch = xt[indices]
            logits = [head(batch) for head in heads]
            rho = 0.1 + 0.9 * math.exp(-5 * (1 - step / total_steps) ** 2)
            total = 0
            for j, values in enumerate(logits):
                solver = solvers[j]
                solver.rho = rho
                if bank is None:
                    target1 = solver(values.detach())
                    target2 = solver(values.detach())
                else:
                    n = len(indices)
                    bank[j] = torch.roll(bank[j], -n, dims=0)
                    bank[j][-n:] = values.detach()
                    target1 = solver(bank[j])[-n:]
                    target2 = solver(bank[j])[-n:]
                assert torch.isfinite(target1).all() and torch.isfinite(target2).all(), (epoch, step, "nonfinite target")
                loss = F.cross_entropy(values, target2) + F.cross_entropy(values, target1)
                total = total + loss
                losses[j] += float(loss.detach()) * len(indices)
            assert torch.isfinite(total), (epoch, step, "nonfinite loss")
            optim.zero_grad(set_to_none=True)
            total.backward()
            optim.step()
            step += 1
        losses /= len(x)
        chosen = int(losses.argmin())
        selection_loss = float(losses[chosen] / rho)
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_head = chosen
            best_state = [{name: tensor.detach().cpu().clone() for name, tensor in h.state_dict().items()} for h in heads]
        history.append({"epoch": epoch + 1, "losses": losses.tolist(), "rho": rho, "lr": lr,
                        "selected_head": chosen, "selection_loss": selection_loss})
        print(json.dumps({"epoch": epoch + 1, "loss": losses.tolist(), "rho": rho}), flush=True)
    heads[best_head].load_state_dict(best_state[best_head])
    with torch.inference_mode():
        pred = torch.cat([heads[best_head](b).argmax(1) for b in xt.split(4096)]).cpu().numpy()
    return pred, {"history": history, "best_head": best_head, "best_loss": best_loss,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "prior_rank": None if prior_rank is None else [r.tolist() for r in prior_rank]}


def fit_mini(x, k, seed, epochs):
    m = 2 * k
    centers = []
    for n in (k, m):
        km = KMeans(n_clusters=n, n_init=10, max_iter=300, tol=1e-4,
                    algorithm="lloyd", random_state=seed)
        km.fit(x)
        centers.append(normalize(km.cluster_centers_).astype(np.float32))
    torch.manual_seed(seed)
    target_head = nn.Linear(x.shape[1], k).cuda()
    mini_head = nn.Linear(x.shape[1], m).cuda()
    with torch.no_grad():
        target_head.weight.copy_(100 * torch.from_numpy(centers[0]).cuda())
        mini_head.weight.copy_(100 * torch.from_numpy(centers[1]).cuda())
        target_head.bias.zero_()
        mini_head.bias.zero_()
    optim = torch.optim.Adam(list(target_head.parameters()) + list(mini_head.parameters()), lr=1e-5)
    xt = torch.from_numpy(x).cuda()
    gen = torch.Generator().manual_seed(seed)
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        perm = torch.randperm(len(x), generator=gen).cuda()
        total_loss, used = 0.0, 0
        for indices in perm.split(512):
            z = xt[indices]
            lt, lm = target_head(z), mini_head(z)
            with torch.no_grad():
                pt, pm = lt.softmax(1), lm.softmax(1)
                ct, yt = pt.max(1)
                cm, ym = pm.max(1)
                table = torch.bincount(yt * m + ym, minlength=k*m).reshape(k,m).float()
                affiliation = F.normalize(table, p=1, dim=0) > 0.3
                weights = m / torch.count_nonzero(affiliation, dim=1).float().clamp_min(0.5)
            mask_t, mask_m = ct > 0.99, cm > 0.99
            if not bool(mask_t.any()) or not bool(mask_m.any()):
                continue
            loss_t = F.cross_entropy(lt[mask_t], yt[mask_t], weight=weights)
            loss_m = F.cross_entropy(lm[mask_m], ym[mask_m])
            st = F.normalize(pt, dim=1) @ F.normalize(pt, dim=1).T
            sm = F.normalize(pm, dim=1) @ F.normalize(pm, dim=1).T
            loss = loss_t + loss_m + F.mse_loss(st, sm)
            assert torch.isfinite(loss)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.detach())
            used += 1
        history.append({"epoch": epoch+1, "loss": total_loss/max(used,1), "used_batches": used})
        print(json.dumps(history[-1]), flush=True)
    with torch.inference_mode():
        pred = torch.cat([target_head(b).argmax(1) for b in xt.split(4096)]).cpu().numpy()
    return pred, {"history": history, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def fit_silhouette(x, seed):
    rng = np.random.RandomState(seed)
    ids = rng.permutation(len(x))[:2000]
    xs = x[ids]
    scores = {}
    models = {}
    for k in range(20, 201, 10):
        model = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4,
                       algorithm="lloyd", random_state=seed).fit(xs)
        scores[k] = float(silhouette_score(xs, model.labels_, metric="euclidean"))
        models[k] = model
        print(json.dumps({"k": k, "silhouette": scores[k]}), flush=True)
    best = min(scores, key=lambda k: (-scores[k], k))
    pred = models[best].predict(x)
    return best, pred, {"scores": scores, "sample_indices_sha256": array_sha(ids)}


def run(args):
    setup(args.seed)
    x, y, stage1 = load_data(args.dataset, args.ir, args.seed)
    ktrue = 10 if args.dataset == "cifar10" else 100
    k = args.k or ktrue
    out = ROOT / args.out / f"{args.dataset}_ir{args.ir}_s{args.seed}" / f"{args.method}_k{k}"
    target = out / "result.json"
    if target.exists():
        raise RuntimeError(f"Refuse to overwrite completed run: {target}")
    start = time.perf_counter()
    if args.method == "KM":
        estimator = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4,
                           algorithm="lloyd", random_state=args.seed)
        pred = estimator.fit_predict(x).astype(np.int64)
        fit = {"inertia": float(estimator.inertia_), "n_iter": int(estimator.n_iter_)}
    elif args.method == "KM_ORACLE_INIT":
        assert k == ktrue
        centers = np.stack([x[y == c].mean(0) for c in range(ktrue)])
        estimator = KMeans(n_clusters=k, init=centers, n_init=1, max_iter=300,
                           tol=1e-4, algorithm="lloyd", random_state=args.seed)
        pred = estimator.fit_predict(x)
        fit = {"inertia": float(estimator.inertia_), "n_iter": int(estimator.n_iter_)}
    elif args.method == "ORACLE_KM":
        assert k == ktrue
        centers = np.stack([x[y == c].mean(0) for c in range(ktrue)])
        pred = (2*x@centers.T - (centers**2).sum(1)).argmax(1).astype(np.int64)
        fit = {"iterations": 0}
    elif args.method in ("P2OT_full", "P2OT_TRUEPRIOR"):
        pred, fit = fit_ppot(x, k, args.seed, args.epochs,
                             np.bincount(y, minlength=ktrue) if args.method == "P2OT_TRUEPRIOR" else None)
    elif args.method == "MiniClustering_feat":
        pred, fit = fit_mini(x, k, args.seed, args.epochs)
    elif args.method == "silhouette":
        k, pred, fit = fit_silhouette(x, args.seed)
    else:
        raise ValueError(args.method)
    measured = evaluate(y, pred, k)
    dump(target, {"dataset": args.dataset, "ir": args.ir, "seed": args.seed, "k": k,
                  "method": args.method, "epochs": args.epochs, "n": len(x),
                  "protocol_sha256": sha(PROTOCOL), "indices_array_sha256": stage1["indices_array_sha256"],
                  "prediction_sha256": array_sha(pred), "metrics": measured, "fit": fit,
                  "elapsed_s": time.perf_counter()-start, "torch": torch.__version__,
                  "code_sha256": sha(Path(__file__))})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--ir", type=int, choices=[1,10,100], required=True)
    p.add_argument("--seed", type=int, choices=[1,2,3,4,5], required=True)
    p.add_argument("--method", choices=["KM","ORACLE_KM","KM_ORACLE_INIT","P2OT_full","P2OT_TRUEPRIOR","MiniClustering_feat","silhouette"], required=True)
    p.add_argument("--k", type=int)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out", default="results_stage2/full")
    a = p.parse_args()
    with threadpool_limits(limits=4):
        run(a)
