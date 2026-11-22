"""Aggregate logged results into results/RESULTS.md, web/results.json and plots. Reads only logged CSVs.

    python -m latent_planner.report --tag main
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette (light mode), fixed order; neutral for baselines.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"

CONFIGS = [  # (run-name prefix, label, ours, color); runs are grouped by name, so variants never mix
    ("lam0.09_s", "Latent CEM, LeWM-style (λ=0.09)", True, BLUE),
    ("fs4_lam0.09_s", "Latent CEM, frameskip 4 (λ=0.09)", True, ORANGE),
    ("ln_fs4_lam0.09_s", "Latent CEM, frameskip 4, LayerNorm projector", True, "#6250d6"),
    ("lam0_s", "Latent CEM, no SIGReg (λ=0)", False, AQUA),
]


def runs_named(df, prefix, col="run"):
    """Rows whose run is exactly <prefix><seed digit> (works for 'runs/x_s0' and bare 'x_s0')."""
    names = df[col].fillna("").map(lambda r: Path(r).name)
    return df[names.str.fullmatch(re.escape(prefix) + r"\d")]
BASELINES = [
    ("oracle_cem", "Oracle CEM: true simulator, straight-line cost"),
    ("greedy", "Straight line to goal (true positions)"),
    ("random", "Random actions"),
]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--results", default="results")
    ap.add_argument("--web", default="web/results.json")
    args = ap.parse_args()
    res = Path(args.results)
    ev = pd.read_csv(res / "eval.csv")
    ev = ev[(ev.taskset == "test") & (ev.tag == args.tag)]
    an = pd.read_csv(res / "analysis.csv") if (res / "analysis.csv").exists() else None
    is_rerun = ev.run.fillna("").str.contains("t4_")
    ev_rerun, ev = ev[is_rerun], ev[~is_rerun]
    if an is not None:
        an_rerun, an = an[an.run.str.startswith("t4_")], an[~an.run.str.startswith("t4_")]
    sweep_lams = sorted(set(ev[(ev.method == "latent_cem") & (ev.frameskip == 1)].lam.dropna()) - {0.0, 0.09})
    m = ["success", "success_same", "success_cross", "mean_steps_success", "sec_per_replan"]

    rows, md = [], []
    for prefix, label, ours, _ in CONFIGS:
        g = runs_named(ev[ev.method == "latent_cem"], prefix).drop_duplicates("run", keep="last")
        if g.empty:
            continue
        mu, sd = g[m].mean(), g[m].std(ddof=0)
        rows.append({"label": label, "ours": ours, "n_seeds": len(g), "success": mu.success, "success_std": sd.success,
                     "success_same": mu.success_same, "success_same_std": sd.success_same,
                     "success_cross": mu.success_cross, "success_cross_std": sd.success_cross,
                     "steps": mu.mean_steps_success, "sec_per_replan": mu.sec_per_replan,
                     "per_seed": g.success.round(3).tolist()})
    g = ev[ev.method == "browser_cem"].tail(1)
    if not g.empty:
        r = g.iloc[0]
        demo = str(r.run).split("(")[-1].rstrip(")") if "(" in str(r.run) else "demo model"
        rows.insert(sum(x["ours"] for x in rows), {"label": f"Same planner in the browser (JS + ONNX Runtime Web, {demo})",
                     "ours": True, "n_seeds": 1, "success": r.success, "success_same": r.success_same,
                     "success_cross": r.success_cross, "steps": r.mean_steps_success,
                     "sec_per_replan": r.sec_per_replan, "per_seed": [round(r.success, 3)]})
    for method, label in BASELINES:
        g = ev[ev.method == method].tail(1)
        if g.empty:
            continue
        r = g.iloc[0]
        rows.append({"label": label, "ours": False, "n_seeds": None, "success": r.success, "success_same": r.success_same,
                     "success_cross": r.success_cross, "steps": r.mean_steps_success,
                     "sec_per_replan": r.sec_per_replan})

    def pct(x, s=None):
        return f"{100 * x:.0f}%" if s is None or np.isnan(s) else f"{100 * x:.0f} ± {100 * s:.0f}%"

    md += ["## Goal reaching (100 fixed test tasks, 50-step budget, success = within 4 px)", "",
           "| Method | Seeds | Success | Same room | Other room | Steps (successes) | Per-seed success |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['n_seeds'] or '-'} | {pct(r['success'], r.get('success_std'))} | "
                  f"{pct(r['success_same'], r.get('success_same_std'))} | "
                  f"{pct(r['success_cross'], r.get('success_cross_std'))} | {r['steps']:.1f} | "
                  f"{', '.join(f'{x:.2f}' for x in r.get('per_seed', [])) or '-'} |")
    md += ["", "Mean ± std over training seeds (one planning seed each). Baselines run once (seeded).", ""]

    if an is not None:
        md += ["## Probes (validation data; diagnostics, never used by the planner)", "",
               "| Config | 16-step rollout error | Embedding std | Effective rank (of 32) | Position probe RMSE (px) | Surprise AUROC | "
               "Latent vs geodesic ρ | Latent vs straight-line ρ | Cost basin radius (px) |", "|---|---|---|---|---|---|---|---|---|"]
        for prefix, label, _, _ in CONFIGS:
            g = runs_named(an, prefix).drop_duplicates("run", keep="last")
            if g.empty:
                continue

            def f(c, d=3):
                return f"{g[c].mean():.{d}f} ± {g[c].std(ddof=0):.{d}f}"

            zs = [json.loads((Path("runs") / r / "summary.json").read_text())["val_z_std"] for r in g.run]
            zstd = f"{np.mean(zs):.4f}" if np.mean(zs) < 0.01 else f"{np.mean(zs):.2f}"
            md.append(f"| {label} | {f('val_nmse_e16')} | {zstd} | {f('val_eff_rank', 1)} | {f('probe_rmse_px', 2)} | "
                      f"{f('surprise_auroc')} | {f('spearman_geo', 2)} | {f('spearman_euc', 2)} | {f('basin_px', 1)} |")
        md += ["", "Rollout error is normalised by the latent variance (1.0 = no better than the mean embedding). "
               "Surprise AUROC separates real next frames from teleports using the prediction error. Effective rank is "
               "scale-invariant, so it stays high for a collapsed model (tiny isotropic noise); embedding std shows "
               "the collapse.", ""]

    if sweep_lams and an is not None:
        md += ["## SIGReg weight sweep (frameskip 1, same test tasks)", "",
               "| λ | Seeds | Success | Other room | Embedding std | 16-step rollout error | Position probe RMSE (px) | "
               "Surprise AUROC | Trained on |", "|---|---|---|---|---|---|---|---|---|"]
        for lam in sorted(set(sweep_lams) | {0.0, 0.09}):
            g = ev[(ev.method == "latent_cem") & (ev.lam == lam) & (ev.frameskip == 1)].drop_duplicates("run", keep="last")
            a_ = an[(an.lam == lam) & (an.frameskip == 1)].drop_duplicates("run", keep="last")
            if g.empty:
                continue
            zs = [json.loads((Path("runs") / Path(r).name / "summary.json").read_text()) for r in g.run]
            dev = sorted({z.get("device", "NVIDIA GeForce RTX 3050 Laptop GPU") for z in zs})
            zstd = np.mean([z["val_z_std"] for z in zs])
            md.append(f"| {lam:g} | {len(g)} | {pct(g.success.mean(), g.success.std(ddof=0))} | "
                      f"{pct(g.success_cross.mean(), g.success_cross.std(ddof=0))} | "
                      f"{zstd:.4f} | {a_.val_nmse_e16.mean():.3f} | {a_.probe_rmse_px.mean():.2f} | "
                      f"{a_.surprise_auroc.mean():.3f} | {', '.join(d.replace('NVIDIA ', '') for d in dev)} |")
        md += ["", "λ = 0.09 is the LeWM default used for the main results. Runs other than λ ∈ {0, 0.09} were trained "
               "and evaluated on a free Kaggle T4 from the same seeds and code (dataset regenerated there; sha256 "
               "matches the local data).", ""]
    if not ev_rerun.empty:
        r = ev_rerun.iloc[-1]
        md += ["## Cross-hardware re-run", "",
               f"λ = 0.09, seed 0, retrained on a Kaggle T4 with the same code and data: success {pct(r.success)} "
               f"(same room {pct(r.success_same)}, other room {pct(r.success_cross)}), {r.mean_steps_success:.1f} "
               "steps; the local RTX 3050 run of the same seed scored 100%.", ""]

    push_rows, push_note = [], ""
    if (res / "eval_push.csv").exists():
        ep = pd.read_csv(res / "eval_push.csv")
        et = ep[(ep.taskset == "test") & (ep.tag == "push")]
        pm = ["success", "success_short", "success_long", "mean_steps_success"]
        for prefix, plabel in [("push_fs4_s", "Latent CEM (λ=0.09, frameskip 4), in the demo"),
                               ("push_ln_fs4_s", "Same, LayerNorm projector"),
                               ("pushv2_ln_s", "LayerNorm, 2× data (90% pushing), 12k steps")]:
          g = runs_named(et[et.method == "latent_cem"], prefix).drop_duplicates("run", keep="last")
          if not g.empty:
            mu, sd = g[pm].mean(), g[pm].std(ddof=0)
            push_rows.append({"label": plabel, "ours": True, "n_seeds": len(g),
                              "success": mu.success, "success_std": sd.success, "success_short": mu.success_short,
                              "success_short_std": sd.success_short, "success_long": mu.success_long,
                              "success_long_std": sd.success_long, "steps": mu.mean_steps_success,
                              "per_seed": g.success.round(3).tolist()})
        for method, label in [("scripted", "Hand-written pusher (true positions)"),
                              ("oracle_cem", "Oracle CEM: true simulator, state-distance cost"),
                              ("random", "Random actions")]:
            b = et[et.method == method].tail(1)
            if not b.empty:
                r = b.iloc[0]
                push_rows.append({"label": label, "ours": False, "n_seeds": None, "success": r.success,
                                  "success_short": r.success_short, "success_long": r.success_long,
                                  "steps": r.mean_steps_success})
        if push_rows:
            push_note = ("100 fixed test tasks; each goal is the end of a 30-step push from the start. Success = block "
                         "within 3 px of its target within 60 steps; long = the block must move 20 px or more. "
                         "Mean ± std over 3 training seeds.")
            md += ["## Scene 2: Push (100 fixed test tasks, 60-step budget, success = block within 3 px)", "",
                   "| Method | Seeds | Success | Short push (< 20 px) | Long push | Steps (successes) | Per-seed |",
                   "|---|---|---|---|---|---|---|"]
            for r in push_rows:
                md.append(f"| {r['label']} | {r['n_seeds'] or '-'} | {pct(r['success'], r.get('success_std'))} | "
                          f"{pct(r['success_short'], r.get('success_short_std'))} | "
                          f"{pct(r['success_long'], r.get('success_long_std'))} | {r['steps']:.1f} | "
                          f"{', '.join(f'{x:.2f}' for x in r.get('per_seed', [])) or '-'} |")
            md += ["", push_note, ""]
        tv = ep[(ep.taskset == "tune") & (ep.method == "latent_cem")].drop_duplicates(["run", "cem"], keep="last")
        if not tv.empty:
            md += ["### Push variants tried on the tuning tasks (seed 0 unless noted)", "",
                   "| Run | Planner | Success | Short | Long | Block read-out error (px) |", "|---|---|---|---|---|---|"]
            for _, r in tv.iterrows():
                sm = json.loads((Path("runs") / Path(r.run).name / "summary.json").read_text())
                md.append(f"| {Path(r.run).name} | {r.cem} | {pct(r.success)} | {pct(r.success_short)} | "
                          f"{pct(r.success_long)} | {sm.get('probe_rmse_block_px', float('nan')):.1f} |")
            md += [""]

    (res / "RESULTS.md").write_text("# Results\n\nGenerated by `python -m latent_planner.report` from "
                                    "`results/eval.csv` and `results/analysis.csv`.\n\n" + "\n".join(md) + "\n")
    note = ("100 fixed test tasks, half needing the door; success = within 4 px of the goal within 50 steps. "
            "Mean ± std over 3 training seeds. Oracle CEM plans with the true simulator but straight-line distance.")
    Path(args.web).write_text(json.dumps({"rows": rows, "note": note, "push_rows": push_rows,
                                          "push_note": push_note}, indent=1))

    # ---- plots
    plots = res / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(rows) + 1.2), sharey=True)
    for ax, key, title in [(axes[0], "success_same", "Same room"), (axes[1], "success_cross", "Other room (through the door)")]:
        y = np.arange(len(rows))[::-1]
        vals = [r[key] * 100 for r in rows]
        errs = [r.get(key + "_std") or 0.0 for r in rows]
        cols = [BLUE if r["ours"] else NEUTRAL for r in rows]
        ax.barh(y, vals, color=cols, height=0.62)
        ax.tick_params(axis="y", length=0)
        for yi, v, e, c in zip(y, vals, errs, cols):
            if e > 0:
                ax.errorbar(v, yi, xerr=e * 100, color=INK2, lw=1, capsize=2)
            if v >= 88:  # label inside the bar end
                ax.text(v - 2, yi, f"{v:.0f}%", va="center", ha="right", fontsize=9,
                        color="white" if c == BLUE else INK)
            else:
                ax.text(v + e * 100 + 2, yi, f"{v:.0f}%", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 105)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style(ax)
    axes[0].set_yticks(np.arange(len(rows))[::-1], [r["label"] for r in rows], fontsize=9, color=INK)
    fig.suptitle("Goal-reaching success on 100 test tasks (blue: ours; error bars: std over 3 training seeds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(plots / "success.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for prefix, label, _, color in CONFIGS:
        logs = sorted(p for p in Path("runs").glob(f"{prefix}*/log.csv") if re.fullmatch(re.escape(prefix) + r"\d", p.parent.name))
        for i, p in enumerate(logs):
            d = pd.read_csv(p)
            col = "val_nmse_e16" if "val_nmse_e16" in d else "val_nmse_h16"
            axes[0].plot(d.step, d[col], color=color, lw=2 if i == 0 else 1, alpha=1 if i == 0 else 0.45,
                         label=label if i == 0 else None)
            axes[1].plot(d.step, d.val_z_std, color=color, lw=2 if i == 0 else 1, alpha=1 if i == 0 else 0.45,
                         label=label if i == 0 else None)
    axes[0].set_yscale("log")
    axes[0].set_title("16-step open-loop rollout error (val, normalised)", fontsize=10, color=INK, loc="left")
    axes[1].set_yscale("log")
    axes[1].set_title("Embedding std per dimension (val); collapse = falls to ~0", fontsize=10, color=INK, loc="left")
    for ax in axes:
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlabel("training step", fontsize=9, color=INK2)
    axes[1].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle("Training curves (bold: seed 0, thin: seeds 1-2)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(plots / "training.png", dpi=150)
    plt.close(fig)
    if sweep_lams:
        lams = sorted(set(sweep_lams) | {0.0, 0.09})
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        xpos = {lam: i for i, lam in enumerate(lams)}
        for lam in lams:
            g = ev[(ev.method == "latent_cem") & (ev.lam == lam) & (ev.frameskip == 1)].drop_duplicates("run", keep="last")
            if g.empty:
                continue
            col = BLUE if lam == 0.09 else NEUTRAL
            ax.bar(xpos[lam], g.success.mean() * 100, color=col, width=0.6)
            ax.scatter([xpos[lam]] * len(g), g.success * 100, color=INK, s=12, zorder=3)
            ax.text(xpos[lam], g.success.max() * 100 + 4, f"{g.success.mean() * 100:.0f}%", ha="center", fontsize=9,
                    color=INK)
        ax.set_xticks(range(len(lams)), [f"{lam:g}" for lam in lams])
        ax.set_xlabel("SIGReg weight λ (blue: LeWM default, used for the main results)", fontsize=9, color=INK2)
        ax.set_ylim(0, 112)
        ax.set_title("Test success vs SIGReg weight (bars: mean, dots: seeds)", fontsize=10, color=INK, loc="left")
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        fig.tight_layout()
        fig.savefig(plots / "sweep.png", dpi=150)
        plt.close(fig)
    print((res / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
