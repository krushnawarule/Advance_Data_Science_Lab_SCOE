"""Assignment 5 - Generative AI: prompt-based text generation with a small language model.
No internet / LLM weights are available, so a small trigram language model 'pre-trained' on the
dataset corpus is used. The mechanism is the same as in an LLM: prompt -> next-token probability
distribution -> temperature / top-k sampling -> append token -> repeat."""
import numpy as np, pandas as pd
from collections import defaultdict, Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

corpus = pd.read_csv("dataset.csv")["text"].tolist()

# --- REPORT START ---
counts = defaultdict(Counter)                           # "pre-training": learn P(next word | last 2 words)
for line in corpus:
    w = ["<s>", "<s>"] + line.split()
    for i in range(2, len(w)): counts[(w[i - 2], w[i - 1])][w[i]] += 1

def next_probs(ctx, temp=1.0, top_k=None):
    c = counts.get(tuple(ctx[-2:])) or counts[("<s>", "<s>")]
    words = list(c); p = np.array([c[x] for x in words], float) ** (1 / temp)   # temperature
    if top_k:                                                                    # top-k filtering
        p[np.argsort(p)[:-top_k]] = 0
    return words, p / p.sum()

def generate(prompt, temp=1.0, top_k=None, max_words=14, seed=0):
    rng = np.random.RandomState(seed); out = prompt.split()
    ctx = ["<s>", "<s>"] + out                           # separate list (not the same object as out)
    for _ in range(max_words):
        words, p = next_probs(ctx, temp, top_k); nxt = rng.choice(words, p=p)
        if nxt == ".": break
        out.append(nxt); ctx.append(nxt)
    return " ".join(out) + " ."
# --- REPORT END ---

if __name__ == "__main__":
    tokens = sum(len(l.split()) for l in corpus)
    print(f"Corpus: {len(corpus)} sentences, {tokens} words, vocabulary={len({w for l in corpus for w in l.split()})}")
    print(f"Learned contexts (trigram states): {len(counts)}")
    print("\n--- Prompt-based generation (temperature = 0.7, top-k = 5) ---")
    for pr in ["data science", "machine learning", "a neural network", "a data scientist", "the model"]:
        print(f"Prompt: '{pr}'\n  -> {generate(pr, 0.7, 5, seed=1)}")
    print("\n--- Effect of temperature on the prompt 'a neural network' ---")
    for t in [0.3, 1.0, 2.0]:
        print(f"T={t}: {generate('a neural network', t, seed=4)}")
    # next-word distribution for one prompt at different temperatures
    ctx = "the model".split(); fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    print("\nNext-word probabilities after 'the model':")
    for a, t in zip(ax, [0.5, 1.0, 2.0]):
        w, p = next_probs(ctx, t); o = np.argsort(-p)[:7]
        a.bar([w[i] for i in o], p[o], color="tab:blue"); a.set_title(f"Temperature = {t}")
        a.tick_params(axis="x", rotation=60)
        print(f"  T={t}: " + ", ".join(f"{w[i]}={p[i]:.2f}" for i in o[:4]))
    ax[0].set_ylabel("Probability"); plt.suptitle("Next-word distribution for prompt: 'the model'")
    plt.tight_layout(); plt.savefig("fig1_next_word_distribution.png", dpi=130); plt.close()
    # perplexity of the model on held-out style sentences (lower = better language model)
    test = ["data science can improve hidden patterns in real world applications .",
            "a transformer is able to extract text data using training data ."]
    lp = n = 0
    for s in test:
        w = ["<s>", "<s>"] + s.split()
        for i in range(2, len(w)):
            c = counts.get((w[i - 2], w[i - 1]), Counter()); tot = sum(c.values())
            lp += np.log((c[w[i]] + 0.1) / (tot + 0.1 * 60)); n += 1
    print(f"\nPerplexity on 2 unseen sentences: {np.exp(-lp / n):.2f}")
