"""Assignment 2 - RNN for sequence prediction (implemented from scratch in NumPy)"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(1)
series = pd.read_csv("dataset.csv")["value"].values
T, H = 20, 16                                         # sequence length, hidden units
Xs = np.array([series[i:i + T] for i in range(len(series) - T)])
ys = series[T:].reshape(-1, 1)                        # target = next value
cut = int(0.8 * len(Xs))
Xtr, ytr, Xte, yte = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]

# --- REPORT START ---
p = {"Wx": np.random.randn(1, H) * 0.5, "Wh": np.random.randn(H, H) * 0.3,
     "b": np.zeros(H), "Wy": np.random.randn(H, 1) * 0.3, "by": np.zeros(1)}

def forward(X):
    hs = [np.zeros((len(X), H))]
    for t in range(X.shape[1]):                       # h_t = tanh(x_t*Wx + h_(t-1)*Wh + b)
        hs.append(np.tanh(X[:, t:t + 1] @ p["Wx"] + hs[-1] @ p["Wh"] + p["b"]))
    return hs[-1] @ p["Wy"] + p["by"], hs             # output from last hidden state

def backward(X, y, out, hs):                          # Backpropagation Through Time (BPTT)
    N = len(X); d = 2 * (out - y) / N
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["Wy"] = hs[-1].T @ d; g["by"] = d.sum(0); dh = d @ p["Wy"].T
    for t in reversed(range(X.shape[1])):
        dpre = dh * (1 - hs[t + 1] ** 2)
        g["Wx"] += X[:, t:t + 1].T @ dpre; g["Wh"] += hs[t].T @ dpre; g["b"] += dpre.sum(0)
        dh = dpre @ p["Wh"].T
    for k in g: g[k] = np.clip(g[k], -1, 1)           # gradient clipping
    return g
# --- REPORT END ---

if __name__ == "__main__":
    print(f"Series length={len(series)} | window={T} | train samples={len(Xtr)} test samples={len(Xte)}")
    print("Trainable parameters:", sum(v.size for v in p.values()))
    m = {k: 0 * v for k, v in p.items()}; v2 = {k: 0 * v for k, v in p.items()}; step = 0; losses = []
    for ep in range(1, 101):
        order = np.random.permutation(len(Xtr))
        for i in range(0, len(Xtr), 32):
            b = order[i:i + 32]; out, hs = forward(Xtr[b]); g = backward(Xtr[b], ytr[b], out, hs); step += 1
            for k in p:                               # Adam update
                m[k] = 0.9 * m[k] + 0.1 * g[k]; v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                p[k] -= 0.01 * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
        losses.append(((forward(Xtr)[0] - ytr) ** 2).mean())
        if ep % 20 == 0 or ep == 1: print(f"Epoch {ep:3d}/100  train MSE = {losses[-1]:.5f}")
    pred = forward(Xte)[0]
    rmse = np.sqrt(((pred - yte) ** 2).mean()); naive = np.sqrt(((Xte[:, -1:] - yte) ** 2).mean())
    ss_res = ((pred - yte) ** 2).sum(); ss_tot = ((yte - yte.mean()) ** 2).sum()
    print(f"\nTest RMSE (RNN)             : {rmse:.4f}")
    print(f"Test RMSE (naive last value): {naive:.4f}")
    print(f"R2 score (RNN)              : {1 - ss_res / ss_tot:.4f}")

    plt.figure(figsize=(9, 3)); plt.plot(losses, c="tab:red"); plt.title("RNN Training Loss (MSE)")
    plt.xlabel("Epoch"); plt.yscale("log"); plt.tight_layout(); plt.savefig("fig1_training_loss.png", dpi=130); plt.close()
    plt.figure(figsize=(9, 3.4)); plt.plot(yte[:150], label="Actual"); plt.plot(pred[:150], "--", label="RNN prediction")
    plt.title("Next-value prediction on test data"); plt.xlabel("Time step"); plt.legend()
    plt.tight_layout(); plt.savefig("fig2_prediction_vs_actual.png", dpi=130); plt.close()
