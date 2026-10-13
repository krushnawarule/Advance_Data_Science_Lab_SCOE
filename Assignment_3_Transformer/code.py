"""Assignment 3 - Transformer encoder for sentiment classification (NumPy, from scratch)"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(3)
df = pd.read_csv("dataset.csv")
vocab = sorted({w for s in df.text for w in s.split()}); w2i = {w: i for i, w in enumerate(vocab)}
ids = np.array([[w2i[w] for w in s.split()] for s in df.text]); y = df.label.values
T, D, FF = ids.shape[1], 16, 32                         # tokens, embedding size, feed-forward size
cut = int(0.8 * len(ids)); tr, te = np.arange(cut), np.arange(cut, len(ids))

# --- REPORT START ---
pos = np.arange(T)[:, None] / 10000 ** (2 * (np.arange(D)[None, :] // 2) / D)
PE = np.where(np.arange(D) % 2 == 0, np.sin(pos), np.cos(pos))       # positional encoding
r = lambda *s: np.random.randn(*s) * 0.2
P = {"E": r(len(vocab), D), "Wq": r(D, D), "Wk": r(D, D), "Wv": r(D, D),
     "W1": r(D, FF), "b1": np.zeros(FF), "W2": r(FF, D), "b2": np.zeros(D), "wc": r(D), "bc": np.zeros(1)}

def forward(tok):
    X = P["E"][tok] + PE                                              # embedding + position
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]
    S = Q @ K.T / np.sqrt(D)                                          # scaled dot-product attention
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    H1 = X + A @ V                                                    # residual connection
    Z = np.maximum(H1 @ P["W1"] + P["b1"], 0)                         # feed-forward network
    H2 = H1 + Z @ P["W2"] + P["b2"]                                   # residual connection
    pooled = H2.mean(0)                                               # average over tokens
    prob = 1 / (1 + np.exp(-(pooled @ P["wc"] + P["bc"][0])))         # sigmoid classifier
    return prob, (tok, X, Q, K, V, A, H1, Z, pooled)

def backward(label, prob, c):
    tok, X, Q, K, V, A, H1, Z, pooled = c; g = {}
    dl = prob - label
    g["wc"] = pooled * dl; g["bc"] = np.array([dl]); dH2 = np.tile(P["wc"] * dl / T, (T, 1))
    g["W2"] = Z.T @ dH2; g["b2"] = dH2.sum(0)
    dpre = (dH2 @ P["W2"].T) * (Z > 0)
    g["W1"] = H1.T @ dpre; g["b1"] = dpre.sum(0); dH1 = dH2 + dpre @ P["W1"].T
    dA = dH1 @ V.T; dV = A.T @ dH1
    dS = A * (dA - (dA * A).sum(1, keepdims=True))
    dQ, dK = dS @ K / np.sqrt(D), dS.T @ Q / np.sqrt(D)
    g["Wq"], g["Wk"], g["Wv"] = X.T @ dQ, X.T @ dK, X.T @ dV
    dX = dH1 + dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T
    g["E"] = np.zeros_like(P["E"]); np.add.at(g["E"], tok, dX)
    return g
# --- REPORT END ---

def accuracy(idx): return np.mean([(forward(ids[i])[0] > 0.5) == y[i] for i in idx])

if __name__ == "__main__":
    # gradient check (numerical vs analytical) to verify the manual backpropagation
    pr, c = forward(ids[0]); g = backward(y[0], pr, c); maxerr = 0
    for k in ["Wq", "Wk", "Wv", "W1", "wc"]:
        i = (0,) * P[k].ndim; old = P[k][i]; L = lambda: -np.log(forward(ids[0])[0] if y[0] else 1 - forward(ids[0])[0])
        P[k][i] = old + 1e-5; a = L(); P[k][i] = old - 1e-5; b = L(); P[k][i] = old
        maxerr = max(maxerr, abs((a - b) / 2e-5 - g[k][i]))
    print(f"Gradient check max error: {maxerr:.2e} (backprop is correct if this is tiny)")
    print(f"Vocabulary={len(vocab)} words | sentences={len(ids)} | train={len(tr)} test={len(te)}")
    print("Trainable parameters:", sum(v.size for v in P.values()))
    m = {k: 0 * v for k, v in P.items()}; v2 = {k: 0 * v for k, v in P.items()}; step = 0; hist = []
    for ep in range(1, 31):
        for i in np.random.permutation(tr):
            pr, c = forward(ids[i]); g = backward(y[i], pr, c); step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]; v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                P[k] -= 0.005 * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
        hist.append((accuracy(tr), accuracy(te)))
        if ep % 5 == 0 or ep == 1: print(f"Epoch {ep:2d}/30  train_acc={hist[-1][0]:.4f}  test_acc={hist[-1][1]:.4f}")
    print(f"\nFinal Test Accuracy: {hist[-1][1] * 100:.2f}%")
    for s in ["the movie was not good and really boring", "this film was really amazing and the story",
              "the plot was not bad but i felt it"]:
        toks = s.split()[:T]; pr, _ = forward(np.array([w2i[w] for w in toks]))
        print(f"'{s}' -> {'Positive' if pr > 0.5 else 'Negative'} ({pr:.2f})")

    plt.figure(figsize=(6, 3.2)); plt.plot([h[0] for h in hist], label="Train"); plt.plot([h[1] for h in hist], label="Test")
    plt.title("Transformer accuracy"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig("fig1_accuracy.png", dpi=130); plt.close()
    # attention heat-map for a test sentence containing a negation
    k = next(i for i in te if "not" in df.text[i].split())
    _, c = forward(ids[k]); words = df.text[k].split()
    plt.figure(figsize=(5, 4.3)); plt.imshow(c[5], cmap="magma"); plt.colorbar(label="attention weight")
    plt.xticks(range(T), words, rotation=60); plt.yticks(range(T), words)
    plt.title("Self-attention map (query row -> key column)"); plt.tight_layout()
    plt.savefig("fig2_attention_heatmap.png", dpi=130); plt.close()
