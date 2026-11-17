"""Render paper figures from results/results.json. Run: python -m ztrag.figures"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5})


def m(x):
    return x["mean"], x["ci95"]


def main(res="results/results.json", out="paper/figs"):
    R = json.loads(Path(res).read_text())
    Path(out).mkdir(parents=True, exist_ok=True)

    # Fig: revocation lag
    rv = R["revocation"]
    deltas = [0.25, 1, 6, 24, 168]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    y = [100 * m(rv[str(d)])[0] for d in deltas]
    e = [100 * m(rv[str(d)])[1] for d in deltas]
    ax.errorbar(deltas, y, yerr=e, color=C[0], marker="o", capsize=2, label="A4 pre-filter, periodic ACL sync")
    ax.plot(deltas, [100 * m(rv["zt"])[0]] * len(deltas), color=C[1], marker="s", ls="--",
            label="A5 zero-trust (live PDP)")
    ax.set_xscale("log")
    ax.set_xticks(deltas)
    ax.set_xticklabels(["15m", "1h", "6h", "24h", "7d"])
    ax.set_xlabel("ACL synchronisation interval $\\Delta$")
    ax.set_ylabel("Leaking queries (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/revocation.pdf")

    # Fig: semantic cache
    ca = R["cache"]
    taus = sorted(ca.keys(), key=float)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
    names = {"naive_global": "Identity-agnostic cache", "per_tenant": "Tenant-partitioned cache",
             "zt_revalidate": "ZT: re-authorise on hit"}
    styles = {"naive_global": ("o", "-"), "per_tenant": ("^", ":"), "zt_revalidate": ("s", "--")}
    for i, v in enumerate(names):
        mk, ls = styles[v]
        for ax, metric in zip(axs, ("leak_rate", "hit_rate")):
            ax.errorbar([float(t) for t in taus], [100 * m(ca[t][v][metric])[0] for t in taus],
                        yerr=[100 * m(ca[t][v][metric])[1] for t in taus], color=C[i], marker=mk, ls=ls,
                        capsize=2, label=names[v])
    axs[0].set_ylabel("Responses leaking (%)")
    axs[1].set_ylabel("Cache hit rate (%)")
    for ax in axs:
        ax.set_xlabel("Similarity threshold $\\tau$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{out}/cache.pdf")

    # Fig: label noise
    ln = R["labelnoise"]
    eps = sorted(ln.keys(), key=float)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    lab = {"A4": "No content check", "A5_chunk": "Chunk-level DLP check", "A5": "Document-level quarantine"}
    sty = {"A4": ("o", "-"), "A5_chunk": ("^", ":"), "A5": ("s", "--")}
    for i, k in enumerate(lab):
        mk, ls = sty[k]
        ax.errorbar([100 * float(x) for x in eps], [100 * m(ln[x][k]["insider"]["exposure"])[0] for x in eps],
                    yerr=[100 * m(ln[x][k]["insider"]["exposure"])[1] for x in eps], color=C[i], marker=mk,
                    ls=ls, capsize=2, label=lab[k])
    ax.set_xlabel("Mislabelling rate $\\varepsilon$ (%)")
    ax.set_ylabel("Insider exposure (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/labelnoise.pdf")
    print("figures written to", out)


if __name__ == "__main__":
    main()


def inflight_figs(res="results/inflight.json", out="paper/figs"):
    R = json.loads(Path(res).read_text())["sim"]
    Path(out).mkdir(parents=True, exist_ok=True)
    # TTL sweep
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ttls = ["0", "60", "300", "900", "3600"]
    for i, (wl, lab, mk, ls) in enumerate([("chat", "Chat (median 4 s)", "o", "-"),
                                            ("agentic", "Agentic (median 60 s)", "s", "--")]):
        ax.errorbar([max(float(t), 10) for t in ttls], [R[wl]["ttl"][t]["mean"] for t in ttls],
                    yerr=[R[wl]["ttl"][t]["ci95"] for t in ttls], color=C[i], marker=mk, ls=ls, capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 60, 300, 900, 3600]); ax.set_xticklabels(["live", "1m", "5m", "15m", "1h"])
    ax.set_xlabel("Entitlement cache TTL at retrieval")
    ax.set_ylabel("Unsafe releases per revocation")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{out}/ttl.pdf")

    # replica lag: non-atomic vs atomic-on-replica vs fenced authority
    lags = ["0.0", "0.5", "2.0", "10.0", "30.0"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    names = {"S2": "Non-atomic recheck (replica)", "S3": "Atomic gate on replica", "S4": "Fenced gate on authority"}
    sty = {"S2": ("o", "-"), "S3": ("^", ":"), "S4": ("s", "--")}
    for ax, wl in zip(axs, ("chat", "agentic")):
        for i, k in enumerate(names):
            mk, ls = sty[k]
            ax.errorbar([float(l) + 0.1 for l in lags], [R[wl]["lag"][l][k]["mean"] for l in lags],
                        yerr=[R[wl]["lag"][l][k]["ci95"] for l in lags], color=C[i], marker=mk, ls=ls, capsize=2,
                        label=names[k])
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.6, 2.1, 10.1, 30.1]); ax.set_xticklabels(["0", "0.5", "2", "10", "30"])
        ax.set_xlabel("Mean policy-replica lag (s)")
        ax.set_title("Chat" if wl == "chat" else "Agentic", fontsize=8.5)
    axs[0].set_ylabel("Unsafe releases per revocation")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f"{out}/lag.pdf")

    # streaming
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    keys = ["none", "10.0", "2.0", "0.5"]
    xl = ["none", "10 s", "2 s", "0.5 s"]
    w = 0.38
    import numpy as _np
    x = _np.arange(len(keys))
    for i, wl in enumerate(("chat", "agentic")):
        ax.bar(x + (i - 0.5) * w * 1.05, [R[wl]["stream"][k]["leaked_tokens_per_event"]["mean"] for k in keys], w,
               yerr=[R[wl]["stream"][k]["leaked_tokens_per_event"]["ci95"] for k in keys], color=C[i],
               label="Chat" if wl == "chat" else "Agentic", capsize=2, edgecolor="white", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(xl)
    ax.set_xlabel("Re-authorisation lease during streaming")
    ax.set_ylabel("Tokens streamed after revocation")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{out}/stream.pdf")
    print("in-flight figures written")


if __name__ == "__main__":
    inflight_figs()
