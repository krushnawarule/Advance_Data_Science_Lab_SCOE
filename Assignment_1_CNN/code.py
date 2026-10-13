"""Assignment 1 - CNN for image classification (implemented from scratch in NumPy)"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
df = pd.read_csv("dataset.csv")
X = df.drop(columns="label").values.reshape(-1, 8, 8) / 16.0
y = df["label"].values
idx = np.random.permutation(len(X)); cut = int(0.8 * len(X))
Xtr, Xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

# --- REPORT START ---
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
# --- REPORT END ---

if __name__ == "__main__":
    print(f"Dataset: {len(X)} images of size 8x8, 10 classes | train={len(Xtr)} test={len(Xte)}")
    model = SmallCNN()
    print("Trainable parameters:", sum(v.size for v in model.p.values()))
    hist = model.fit(Xtr, ytr, epochs=30, val=(Xte, yte))
    pred = model.predict(Xte)
    print(f"\nFinal Test Accuracy: {(pred == yte).mean() * 100:.2f}%\n")
    print(classification_report(yte, pred, digits=3))

    # Figure 1: training curves
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(hist["loss"], c="tab:red"); ax[0].set_title("Training Loss"); ax[0].set_xlabel("Epoch")
    ax[1].plot(hist["acc"], label="Train"); ax[1].plot(hist["val"], label="Test")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig("fig1_training_curves.png", dpi=130); plt.close()

    # Figure 2: confusion matrix
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4.8, 4.2)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
    for i in range(10):
        for j in range(10): plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    plt.title("Confusion Matrix (Test)"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(range(10)); plt.yticks(range(10)); plt.tight_layout()
    plt.savefig("fig2_confusion_matrix.png", dpi=130); plt.close()

    # Figure 3: learned filters and feature maps of one image
    _, c = model.forward(Xte[:1]); act, pool = c[2][0], c[4][0]
    fig, ax = plt.subplots(3, 8, figsize=(10, 4.2))
    for f in range(8):
        ax[0, f].imshow(model.p["K"][f], cmap="gray"); ax[0, f].set_title(f"Filter {f+1}", fontsize=7)
        ax[1, f].imshow(act[f], cmap="viridis"); ax[2, f].imshow(pool[f], cmap="viridis")
    ax[1, 0].set_ylabel("After ReLU", fontsize=7); ax[2, 0].set_ylabel("After Pool", fontsize=7)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"Filters & feature maps (input digit = {yte[0]})", fontsize=9)
    plt.tight_layout(); plt.savefig("fig3_filters_feature_maps.png", dpi=130); plt.close()
