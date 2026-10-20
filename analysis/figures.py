"""Journal figures from saved results (TIFF at 600 dpi, PDF, and PNG preview).

Open source libraries: Matplotlib and seaborn for charts, Graphviz for the
study workflow diagram.
"""

from __future__ import annotations

import json
from pathlib import Path

import graphviz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COHORTS = {
    "covid": "COVID-19 collection",
    "lidc": "LIDC-IDRI",
    "cbisddsm": "CBIS-DDSM",
    "nlst": "NLST",
    "nih": "NIH ChestX-ray14",
    "padchest": "PadChest",
}
METHODS = ["Record random", "Group random", "Stratified group", "Rerandomization", "RadSplit"]
COLORS = {
    "Record random": "#8a8a86",
    "Group random": "#eb6834",
    "Stratified group": "#1baf7a",
    "Rerandomization": "#4a3aa7",
    "RadSplit": "#2a78d6",
}
MARKERS = {"Record random": "X", "Group random": "o", "Stratified group": "s",
           "Rerandomization": "D", "RadSplit": "^"}
INK = "#1b1f24"
MUTED = "#5b6168"
GRID = "#d9dcdf"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    }
)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(FIGURES / f"{name}.png").convert("RGB").save(
        FIGURES / f"{name}.tif", dpi=(600, 600), compression="tiff_lzw")


def panel(ax, letter: str) -> None:
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=INK, ha="right", va="bottom")


def plain(value: float, _=None) -> str:
    return f"{value:g}"


def benchmark() -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(RESULTS / f"benchmark{key}.csv") for key in COHORTS],
                      ignore_index=True)
    return frame


def strip_box(ax, data: pd.DataFrame, value: str, order: list[str], log: bool = False) -> None:
    for position, method in enumerate(order):
        values = data.loc[data["method"] == method, value].to_numpy()
        rng = np.random.default_rng(position)
        ax.scatter(position + rng.uniform(-0.2, 0.2, len(values)), values, s=4,
                   color=COLORS[method], alpha=0.35, linewidths=0, zorder=2)
        q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
        ax.plot([position - 0.3, position + 0.3], [median, median], color=INK, lw=1.3, zorder=4)
        ax.plot([position, position], [q1, q3], color=INK, lw=1.0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([m.replace(" ", "\n").replace("Rerandomization", "Rerandom-\nization")
                        for m in order])
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def figure1() -> None:
    """Schematic of the tool: inputs, functions, and outputs."""
    g = graphviz.Digraph("radsplit", format="png")
    g.attr(rankdir="LR", dpi="600", nodesep="0.22", ranksep="0.55", fontname="Liberation Sans",
           bgcolor="white", compound="true")
    g.attr("node", shape="box", style="rounded,filled", fontname="Liberation Sans",
           fontsize="10", color="#5b6168", penwidth="0.9", margin="0.14,0.09")
    g.attr("edge", color="#5b6168", arrowsize="0.7", penwidth="0.9")

    def lines(*text: str) -> str:
        return "<" + "<BR/>".join(text) + ">"

    with g.subgraph(name="cluster_in") as c:
        c.attr(label="Input", fontname="Liberation Sans", fontsize="10.5", color="#b8bdc2",
               style="rounded", labeljust="l")
        c.node("table", lines("<B>Record table</B>", "CSV or Parquet,", "one row per image,",
                              "series, or lesion"), fillcolor="#eef3f9")
        c.node("headers", lines("<B>Image folder</B>", "DICOM, NIfTI,", "MetaImage, NRRD",
                                "headers read into", "a series table"), fillcolor="#eef3f9")
        c.node("labels", lines("<B>Label table</B>", "joined by patient", "or series"),
               fillcolor="#eef3f9")
    with g.subgraph(name="cluster_fn") as c:
        c.attr(label="RadSplit", fontname="Liberation Sans", fontsize="10.5", color="#b8bdc2",
               style="rounded", labeljust="l")
        c.node("audit", lines("<B>Audit</B>", "any existing partition:", "patient overlap, sizes,",
                              "covariate balance"), fillcolor="#fbf1e6")
        c.node("split", lines("<B>Balanced split</B>", "patient disjoint,", "exact sizes,",
                              "balanced labels and", "covariates; folds"), fillcolor="#eaf6f0")
        c.node("hold", lines("<B>Held out group</B>", "scanner, kernel, site,", "or visit as test set;",
                             "leave one group out"), fillcolor="#f1ecf9")
    with g.subgraph(name="cluster_out") as c:
        c.attr(label="Output", fontname="Liberation Sans", fontsize="10.5", color="#b8bdc2",
               style="rounded", labeljust="l")
        c.node("assign", lines("<B>Assignment</B>", "per patient, with hash,", "seed, and configuration"),
               fillcolor="#ffffff")
        c.node("report", lines("<B>Report</B>", "characteristics by partition,",
                               "standardized differences,", "CLAIM items 19 and 20"),
               fillcolor="#ffffff")
        c.node("export", lines("<B>Framework files</B>", "nnU-Net folds, MONAI datalist,",
                               "PyTorch indices,", "scikit-learn splitter"), fillcolor="#ffffff")
    g.edge("headers", "split")
    g.edge("labels", "split")
    g.edge("table", "audit")
    g.edge("table", "split")
    g.edge("table", "hold")
    g.edge("audit", "report")
    g.edge("split", "assign")
    g.edge("split", "report")
    g.edge("hold", "assign")
    g.edge("hold", "export")
    g.edge("split", "export")
    FIGURES.mkdir(exist_ok=True)
    out = Path(g.render(str(FIGURES / "figure1"), cleanup=True))
    Image.MAX_IMAGE_PIXELS = None
    Image.open(out).convert("RGB").save(FIGURES / "figure1.tif", dpi=(600, 600),
                                        compression="tiff_lzw")
    g.format = "pdf"
    g.render(str(FIGURES / "figure1"), cleanup=True)


def figure2(bench: pd.DataFrame) -> None:
    """Patient overlap under record level random partitioning and in distributed partitions."""
    official = json.loads((RESULTS / "officialaudit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    random = bench[bench["method"] == "Record random"].copy()
    random["share"] = random["test_records_patient_in_train"] / random["test_records"]
    for position, key in enumerate(COHORTS):
        values = random.loc[random["cohort"] == key, "share"].to_numpy()
        rng = np.random.default_rng(position)
        ax.scatter(values, position + rng.uniform(-0.18, 0.18, len(values)), s=5,
                   color=COLORS["Record random"], alpha=0.45, linewidths=0, zorder=2)
        median = np.median(values)
        ax.plot([median, median], [position - 0.3, position + 0.3], color=INK, lw=1.4, zorder=3)
        ax.text(median, position - 0.38, f"{100 * median:.0f}%", ha="center", va="bottom",
                fontsize=7.5, color=INK)
    ax.set_yticks(range(len(COHORTS)))
    ax.set_yticklabels(list(COHORTS.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Test records whose patient is also\nin the training partition")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Record level random partitioning", loc="left")
    panel(ax, "A")

    ax = axes[1]
    frames = []
    for key, label in (("nih", "NIH ChestX-ray14"), ("cbisddsm", "CBIS-DDSM")):
        table = pd.read_csv(RESULTS / f"officialbalance{key}.csv")
        table["cohort"] = label
        frames.append(table.sort_values("max_abs_smd", ascending=False).head(7))
    table = pd.concat(frames, ignore_index=True)
    table["name"] = table["term"].map(pretty_term)
    y = 0
    ticks, names = [], []
    for label, color in (("NIH ChestX-ray14", "#4a3aa7"), ("CBIS-DDSM", "#eb6834")):
        part = table[table["cohort"] == label]
        ax.text(0.99, y - 0.75, label, fontsize=7.5, fontweight="bold", color=INK,
                transform=ax.get_yaxis_transform(), ha="right", va="center")
        for _, row in part.iterrows():
            ax.plot([0, row["max_abs_smd"]], [y, y], color=color, lw=1.0, zorder=2)
            ax.scatter(row["max_abs_smd"], y, s=20, color=color, zorder=3)
            ticks.append(y)
            names.append(row["name"])
            y += 1
        y += 1.2
    ax.axvline(0.1, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names)
    ax.set_ylim(y - 1.6, -1.4)
    ax.set_xlim(0, None)
    ax.set_xlabel("Absolute standardized mean difference,\nofficial training versus test set")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Distributed partitions", loc="left")
    panel(ax, "B")
    fig.tight_layout(w_pad=2.0)
    save(fig, "figure6")
    del official


def pretty_term(term: str) -> str:
    names = {
        "records_per_patient": "Radiographs per patient",
        "any_finding: 1": "Any finding",
        "view: PA": "Posteroanterior view",
        "sex: M": "Male sex",
        "age": "Age",
        "abnormality: mass": "Mass (versus calcification)",
        "malignant: 1": "Malignant",
        "suspicious_nodule: 1": "Suspicious nodule",
        "nodule: 1": "Any annotated nodule",
        "contrast: 1": "Contrast enhanced",
        "view: MLO": "Mediolateral oblique view",
        "laterality: RIGHT": "Right breast",
    }
    if term in names:
        return names[term]
    variable, _, level = term.partition(": ")
    variable = variable.replace("_", " ")
    if variable == "assessment":
        return f"BI-RADS assessment {level}"
    if variable == "slice thickness":
        return f"Slice thickness {level}"
    if variable == "kvp":
        return f"Tube voltage {level} kV"
    if variable == "study year":
        relative = level in ("1999", "2000", "2001")  # de-identified dates in the NLST index
        return f"Study year {int(level) - 1998}" if relative else f"Study year {level}"
    if variable in ("breast density", "subtlety"):
        return f"{variable.capitalize()} {level}"
    if level == "1":
        return variable.capitalize()
    return f"{variable.capitalize()} {level}"


def figure3(bench: pd.DataFrame) -> None:
    """Covariate balance across methods and cohorts."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), sharey=True)
    counts = json.loads((RESULTS / "summary.json").read_text())["cohorts"]
    for ax, (key, name) in zip(axes.ravel(), COHORTS.items()):
        strip_box(ax, bench[bench["cohort"] == key], "mean_abs_smd", METHODS, log=True)
        ax.set_title(f"{name} ({counts[key]['patients']:,} patients)", loc="center")
        ax.tick_params(axis="x", labelsize=6.3)
        ax.axhline(0.1, color=MUTED, ls="--", lw=0.8, zorder=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean absolute SMD")
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, "figure3")


def figure4() -> None:
    """Imbalance and run time by cohort size."""
    data = pd.read_csv(RESULTS / "cohortsize.csv")
    order = METHODS[1:]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    specs = [("mean_abs_smd", "Mean absolute SMD"), ("max_abs_smd", "Maximum absolute SMD"),
             ("runtime_seconds", "Run time (seconds)")]
    for ax, (value, label), letter in zip(axes, specs, "ABC"):
        for method in order:
            part = data[data["method"] == method].groupby("patients")[value]
            median, q1, q3 = part.median(), part.quantile(0.25), part.quantile(0.75)
            ax.fill_between(median.index, q1, q3, color=COLORS[method], alpha=0.15, lw=0)
            ax.plot(median.index, median, color=COLORS[method], lw=1.5, marker=MARKERS[method],
                    ms=3.5, label=method)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
        ax.set_xlabel("Patients in cohort")
        ax.set_ylabel(label)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        if value == "max_abs_smd":
            ax.axhline(0.1, color=MUTED, ls="--", lw=0.8)
        panel(ax, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.2)
    save(fig, "figure4")


def downstream_figure() -> None:
    """Downstream model evaluation."""
    tasks = {"covid": "COVID-19 versus other findings\n(radiograph embeddings)",
             "cbisddsm": "Malignant versus benign\n(lesion descriptors)"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), gridspec_kw={"width_ratios": [1.5, 0.8, 1.5]})
    for row, (task, title) in enumerate(tasks.items()):
        data = pd.read_csv(RESULTS / f"downstream{task}.csv")
        ax = axes[row, 0]
        strip_box(ax, data, "auc_test", METHODS)
        ax.set_ylabel("Test AUC")
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", labelsize=6.5)
        panel(ax, "AD"[row])

        ax = axes[row, 1]
        random = data[data["method"] == "Record random"]
        columns = ["auc_test_patient_in_train", "auc_test_patient_not_in_train"]
        for position, column in enumerate(columns):
            values = random[column].to_numpy()
            rng = np.random.default_rng(position)
            ax.scatter(position + rng.uniform(-0.2, 0.2, len(values)), values, s=4,
                       color=COLORS["Record random"], alpha=0.45, linewidths=0)
            q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
            ax.plot([position - 0.3, position + 0.3], [median, median], color=INK, lw=1.3)
            ax.plot([position, position], [q1, q3], color=INK, lw=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Patient in\ntraining", "Patient not\nin training"], fontsize=6.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("Test AUC, record random")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        panel(ax, "BE"[row])

        ax = axes[row, 2]
        data = data.assign(gap=(data["auc_tune"] - data["auc_test"]).abs())
        strip_box(ax, data, "gap", METHODS)
        ax.set_ylabel("Absolute tuning to test AUC difference")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=6.5)
        panel(ax, "CF"[row])
    fig.tight_layout(h_pad=1.6, w_pad=1.0)
    save(fig, "figureS1")


def example_audit() -> None:
    """Example audit output: per term balance for one seed, CBIS-DDSM."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import cohorts
    import methods

    from radsplit import audit

    cohort = cohorts.load("cbisddsm")
    records = cohort.records
    codes = {
        "Group random": methods.group_random(records["patient_id"].to_numpy(), 0),
        "Stratified group": methods.stratified_group(
            records["patient_id"].to_numpy(), records["malignant"].to_numpy(), 0),
        "RadSplit": methods.radsplit_codes(records, cohort.variables, 0),
    }
    tables = {}
    for method, code in codes.items():
        frame = records.assign(split=np.asarray(methods.NAMES)[code])
        tables[method] = audit(frame, patient="patient_id", split="split",
                               variables=cohort.variables,
                               partitions=methods.NAMES).balance.set_index("term")["max_abs_smd"]
    table = pd.DataFrame(tables)
    table = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    y = np.arange(len(table))
    ax.hlines(y, table.min(axis=1), table.max(axis=1), color=GRID, lw=1.0, zorder=1)
    for method in codes:
        ax.scatter(table[method], y, s=22, color=COLORS[method], marker=MARKERS[method],
                   label=method, zorder=3, edgecolor="white", linewidths=0.4)
    ax.axvline(0.1, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_term(t) for t in table.index])
    ax.set_xlim(left=0)
    ax.set_xlabel("Largest absolute SMD between any two partitions (seed 0)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "figureS2")
    table.iloc[::-1].to_csv(RESULTS / "exampleaudit.csv")


def shift_figure() -> None:
    """Shift between development and held out test partitions."""
    terms = pd.read_csv(RESULTS / "shiftterms.csv")
    part = terms[terms["held_out_variable"] == "manufacturer"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, key, letter in zip(axes, ["lidc", "nlst"], "AB"):
        table = part[part["cohort"] == key].pivot(index="term", columns="held_out_value",
                                                  values="smd")
        table = table.loc[table.abs().max(axis=1).sort_values(ascending=False).index[:12]]
        table.index = [pretty_term(t) for t in table.index]
        sns.heatmap(table, ax=ax, cmap="RdBu_r", center=0, vmin=-2, vmax=2, annot=True,
                    fmt=".1f", annot_kws={"size": 6.5}, linewidths=1.0, linecolor="white",
                    cbar=key == "nlst",
                    cbar_kws={"label": "SMD, test minus training", "shrink": 0.8})
        ax.set_xlabel("Held out manufacturer")
        ax.set_ylabel("")
        ax.set_title(COHORTS[key], loc="left")
        ax.tick_params(axis="both", length=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        panel(ax, letter)
    fig.tight_layout(w_pad=1.5)
    save(fig, "figure5")


def usage_figure() -> None:
    """Example calls and the report returned for one of them."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import cohorts

    from radsplit import split

    cohort = cohorts.load("cbisddsm")
    result = split(cohort.records, patient="patient_id", target="malignant",
                   balance=[v for v in cohort.variables if v != "malignant"], seed=0)
    table = result.audit.table().head(6)
    code = "\n".join([
        "from radsplit import audit, split, holdout, leave_one_group_out",
        "",
        "# 1. Audit a partition that already exists",
        'report = audit(table, patient="patient_id", split="official_split",',
        '               variables=["malignant", "breast_density", "view"])',
        "",
        "# 2. Balanced, patient disjoint training, tuning, and test partitions",
        'result = split(table, patient="patient_id", target="malignant",',
        '               balance=["abnormality", "breast_density", "assessment",',
        '                        "subtlety", "view", "laterality"], seed=0)',
        "",
        "# 3. Hold out a scanner, kernel, site, or visit as the test set",
        'shift = holdout(series, patient="patient_id", by="convolution_kernel",',
        '                values=["B70f"], balance=["slice_thickness", "kvp"])',
        "",
        "# 4. Rotate over all scanners for an ablation",
        "for model, fold in leave_one_group_out(series, patient=\"patient_id\",",
        '                                       by="scanner_model"):',
        '    train, tune, test = fold["train"], fold["tune"], fold["test"]',
    ])
    widths = [max(len(str(v)) for v in [c, *table[c]]) for c in table.columns]
    rows = ["  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                      for i, (c, w) in enumerate(zip(table.columns, widths)))]
    for _, r in table.iterrows():
        rows.append("  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w)
                              for i, (v, w) in enumerate(zip(r, widths))))
    out = result.report() + "\n\nCharacteristics by partition (first rows)\n" + "\n".join(rows)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), gridspec_kw={"height_ratios": [1.0, 1.15]})
    for ax, text, title, letter in zip(axes, [code, out], ["Calls", "Output of call 2 (CBIS-DDSM)"], "AB"):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f6f7f8",
                                   edgecolor="#c9cdd2", lw=0.8))
        ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                family="DejaVu Sans Mono", fontsize=6.6, color=INK, linespacing=1.35)
        ax.set_title(title, loc="left")
        panel(ax, letter)
    fig.tight_layout(h_pad=1.5)
    save(fig, "figure2")


def main() -> None:
    bench = benchmark()
    figure1()
    usage_figure()
    figure2(bench)
    figure3(bench)
    figure4()
    shift_figure()
    example_audit()
    downstream_figure()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
