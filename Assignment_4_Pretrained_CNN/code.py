"""Assignment 4 - Transfer learning with a pre-trained CNN (NumPy, from scratch)
Source task (acts like ImageNet): digits 0-4 with LOTS of data -> CNN is pre-trained.
Target task (new dataset)       : digits 5-9 with VERY FEW labelled images.
Pre-trained convolution layers are frozen (feature extractor); only the new fully connected layers are trained."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view as swv

df = pd.read_csv("dataset.csv")
X = df.drop(columns="label").values.reshape(-1, 8, 8) / 16.0
y = df["label"].values
np.random.seed(7)
idx = np.random.permutation(len(X)); cut = int(0.8 * len(X))
tr, te = idx[:cut], idx[cut:]
src_tr = tr[y[tr] < 5]                                   # source: classes 0-4
tgt_tr, tgt_te = tr[y[tr] >= 5], te[y[te] >= 5]          # target: classes 5-9

# CNN class (same architecture as Assignment 1)
class SmallCNN:
    """Conv(3x3,8) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(32)+ReLU -> Dense(10)+Softmax"""
    def __init__(self, nf=8, nclass=10):
        r = np.random.randn; self.nf = nf
        self.p = {"K": r(nf, 3, 3) * 0.3, "kb": np.zeros(nf),
                  "W1": r(nf * 9, 32) * np.sqrt(2 / (nf * 9)), "b1": np.zeros(32),
                  "W2": r(32, nclass) * np.sqrt(2 / 32), "b2": np.zeros(nclass)}

    def forward(self, X):
        p = self.p; N = len(X)
        win = swv(X, (3, 3), axis=(1, 2))                                  # sliding 3x3 windows
        conv = np.einsum("nijkl,fkl->nfij", win, p["K"]) + p["kb"][None, :, None, None]
        act = np.maximum(conv, 0)                                          # ReLU activation
        blk = act.reshape(N, self.nf, 3, 2, 3, 2)
        pool = blk.max(axis=(3, 5))                                        # 2x2 max pooling
        flat = pool.reshape(N, -1)                                         # flatten
        pre = flat @ p["W1"] + p["b1"]; h = np.maximum(pre, 0)             # fully connected + ReLU
        z = h @ p["W2"] + p["b2"]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), (win, conv, act, blk, pool, flat, pre, h)

    def backward(self, y, P, c):
        p = self.p; win, conv, act, blk, pool, flat, pre, h = c; N = len(y)
        d = P.copy(); d[np.arange(N), y] -= 1; d /= N                      # softmax + cross-entropy
        g = {"W2": h.T @ d, "b2": d.sum(0)}
        dh = (d @ p["W2"].T) * (pre > 0)
        g["W1"] = flat.T @ dh; g["b1"] = dh.sum(0)
        dpool = (dh @ p["W1"].T).reshape(pool.shape)
        mask = (blk == pool[:, :, :, None, :, None]).astype(float)
        mask /= mask.sum(axis=(3, 5), keepdims=True)
        dact = (mask * dpool[:, :, :, None, :, None]).reshape(N, self.nf, 6, 6)
        dconv = dact * (conv > 0)
        g["K"] = np.einsum("nijkl,nfij->fkl", win, dconv); g["kb"] = dconv.sum((0, 2, 3))
        return g

    def fit(self, X, y, epochs=30, bs=32, lr=0.01, frozen=(), val=None, log=True):
        st = {"t": 0, "m": {k: 0 * v for k, v in self.p.items()}, "v": {k: 0 * v for k, v in self.p.items()}}
        hist = {"loss": [], "acc": [], "val": []}
        for ep in range(1, epochs + 1):
            order = np.random.permutation(len(X))
            for i in range(0, len(X), bs):
                b = order[i:i + bs]; P, c = self.forward(X[b]); g = self.backward(y[b], P, c)
                st["t"] += 1
                for k in self.p:
                    if k in frozen: continue                               # frozen layers are not updated
                    st["m"][k] = 0.9 * st["m"][k] + 0.1 * g[k]
                    st["v"][k] = 0.999 * st["v"][k] + 0.001 * g[k] ** 2
                    mh = st["m"][k] / (1 - 0.9 ** st["t"]); vh = st["v"][k] / (1 - 0.999 ** st["t"])
                    self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            P, _ = self.forward(X)
            hist["loss"].append(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())
            hist["acc"].append((P.argmax(1) == y).mean())
            if val is not None: hist["val"].append((self.predict(val[0]) == val[1]).mean())
            if log and (ep % 5 == 0 or ep == 1):
                print(f"Epoch {ep:2d}/{epochs}  loss={hist['loss'][-1]:.4f}  train_acc={hist['acc'][-1]:.4f}"
                      + (f"  test_acc={hist['val'][-1]:.4f}" if val is not None else ""))
        return hist

    def predict(self, X): return self.forward(X)[0].argmax(1)

def target_data(n):                                       # small labelled target set (balanced-ish)
    sel = np.random.permutation(tgt_tr)[:n]; return X[sel], y[sel] - 5

if __name__ == "__main__":
    print(f"Source task: {len(src_tr)} images (digits 0-4) | Target task: train pool={len(tgt_tr)}, test={len(tgt_te)} (digits 5-9)")
    # --- REPORT START ---
    print("Step 1: pre-training CNN on the source task ...")
    base = SmallCNN(nclass=5); base.fit(X[src_tr], y[src_tr], epochs=20, log=False)
    print(f"  Source-task training accuracy: {(base.predict(X[src_tr]) == y[src_tr]).mean() * 100:.2f}%")

    def transfer(n):                                      # copy pre-trained conv filters, freeze them,
        m = SmallCNN(nclass=5)                            # train only the new fully connected layers
        for k in ["K", "kb"]: m.p[k] = base.p[k].copy()
        xs, ys = target_data(n); m.fit(xs, ys, epochs=30, frozen=("K", "kb"), log=False)
        return (m.predict(X[tgt_te]) == y[tgt_te] - 5).mean()

    def scratch(n):                                       # same model, random weights, all layers trained
        m = SmallCNN(nclass=5); xs, ys = target_data(n); m.fit(xs, ys, epochs=30, log=False)
        return (m.predict(X[tgt_te]) == y[tgt_te] - 5).mean()
    # --- REPORT END ---
    sizes = [10, 25, 50, 100, 200]; res = {"Pre-trained (frozen conv)": [], "Trained from scratch": []}
    print("\nStep 2: target-task test accuracy (average of 10 runs)")
    print(f"{'Target images':>14} | {'Pre-trained':>12} | {'From scratch':>12}")
    for n in sizes:
        a = np.mean([transfer(n) for _ in range(10)]); b = np.mean([scratch(n) for _ in range(10)])
        res["Pre-trained (frozen conv)"].append(a); res["Trained from scratch"].append(b)
        print(f"{n:>14} | {a * 100:>11.2f}% | {b * 100:>11.2f}%")
    plt.figure(figsize=(6.4, 3.8))
    for k, v in res.items(): plt.plot(sizes, [a * 100 for a in v], marker="o", label=k)
    plt.xlabel("Number of labelled target images"); plt.ylabel("Test accuracy (%)")
    plt.title("Transfer learning vs training from scratch"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig("fig1_transfer_vs_scratch.png", dpi=130); plt.close()
    # feature-map comparison of one target image through pre-trained filters
    fig, ax = plt.subplots(2, 8, figsize=(10, 2.8)); _, c = base.forward(X[tgt_te[:1]])
    for f in range(8):
        ax[0, f].imshow(base.p["K"][f], cmap="gray"); ax[1, f].imshow(c[2][0][f], cmap="viridis")
    ax[0, 0].set_ylabel("Pre-trained filters", fontsize=7); ax[1, 0].set_ylabel("Feature maps", fontsize=7)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"Source-task filters applied to an unseen target digit ({y[tgt_te[0]]})", fontsize=9)
    plt.tight_layout(); plt.savefig("fig2_pretrained_filters.png", dpi=130); plt.close()
