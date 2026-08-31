"""Create plots from the model output tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
PLOTS_DIR = ROOT / "outputs" / "plots"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(PLOTS_DIR / name)
    plt.close(fig)


# Load the data and model outputs.
df = pd.read_csv(ROOT / "data" / "final_shop_6modata.csv")
parts = df["Ad Group"].str.split(" - ", expand=True)
df["match_type"] = parts[1]
df["device"] = parts[2]
df["keyword_theme"] = parts[3].str.replace("[", "", regex=False).str.replace("]", "", regex=False)

monthly = pd.read_csv(METRICS_DIR / "monthly_summary.csv")
ad_group = pd.read_csv(METRICS_DIR / "ad_group_summary.csv")
comparison = pd.read_csv(METRICS_DIR / "allocation_comparison.csv")
sensitivity = pd.read_csv(METRICS_DIR / "allocation_sensitivity.csv")
tiered_sens = pd.read_csv(METRICS_DIR / "tiered_marginal_sensitivity.csv")
backtest = pd.read_csv(METRICS_DIR / "pseudo_backtest.csv")
backtest_floor = pd.read_csv(METRICS_DIR / "pseudo_backtest_floor.csv")
pareto = pd.read_csv(METRICS_DIR / "pareto_frontier.csv")


# Figure 1: funnel volumes
funnel_values = [
    df["Impressions"].sum(),
    df["Clicks"].sum(),
    df["Conversions"].sum(),
]
funnel_labels = ["Impressions", "Clicks", "Conversions"]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
bars = ax.bar(funnel_labels, funnel_values, color=["#4C78A8", "#F28E2B", "#59A14F"])
ax.set_yscale("log")
ax.set_ylabel("count (log scale)")
ax.set_title("Paid-search funnel volumes across the five months")
for bar, val in zip(bars, funnel_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
            f"{val:,.0f}", ha="center", va="bottom", fontsize=11)
ax.set_ylim(top=max(funnel_values) * 3)
plt.tight_layout()
save(fig, "fig1_funnel_totals.png")


# Figure 2: monthly P&L
month_order = ["July", "August", "September", "October", "November"]
monthly_plot = monthly.set_index("Month").loc[month_order].reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
colors = ["#59A14F" if v >= 0 else "#E15759" for v in monthly_plot["profit"]]
bars = ax.bar(monthly_plot["Month"], monthly_plot["profit"], color=colors)
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel(r"monthly P&L (\$)")
ax.set_title("Monthly profit and loss")
ymin = monthly_plot["profit"].min()
ymax = monthly_plot["profit"].max()
ax.set_ylim(ymin * 1.18, max(ymax * 2.5, 3000))
for bar, val in zip(bars, monthly_plot["profit"]):
    inside_y = val * 0.5
    ax.text(bar.get_x() + bar.get_width() / 2, inside_y,
            (r"\$" + f"{val:,.0f}"),
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")
plt.tight_layout()
save(fig, "fig2_monthly_profit.png")


# Figure 3: profit per dollar by device and keyword theme
def summarize_segment(d, group_cols):
    g = d.groupby(group_cols, as_index=False).agg(cost=("Cost", "sum"), profit=("P&L", "sum"))
    g["profit_per_dollar"] = g["profit"] / g["cost"]
    return g

device_theme = summarize_segment(df, ["device", "keyword_theme"])
device_theme = device_theme[device_theme["keyword_theme"] != "Black Friday/Cyber Monday"]
heat = device_theme.pivot(index="keyword_theme", columns="device", values="profit_per_dollar")
heat = heat.sort_values("Desk", ascending=True)

fig, ax = plt.subplots(figsize=(7.0, 6.5))
im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.0)
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_title(
    "Profit per dollar by device and keyword theme\n"
    "(Black Friday/Cyber Monday excluded as low-cost outlier)"
)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        if pd.notna(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    color="white" if v < -0.10 else "black", fontsize=10)
fig.colorbar(im, ax=ax, label="profit per dollar")
plt.tight_layout()
save(fig, "fig3_profit_by_device_theme.png")


# Figure 4: raw and smoothed rates
fig, ax = plt.subplots(figsize=(7.0, 5.0))
sizes = 30 + 250 * (ad_group["cost"] / ad_group["cost"].max())
ax.scatter(ad_group["profit_per_dollar"], ad_group["smoothed_profit_per_dollar"],
           s=sizes, alpha=0.7, color="#4C78A8", edgecolor="white", linewidth=0.7)
ax.plot([-1, 1.6], [-1, 1.6], color="black", linestyle="--", linewidth=1, label="raw = shrunk")
overall_rate = ad_group["profit"].sum() / ad_group["cost"].sum()
ax.axhline(
    overall_rate,
    color="#E15759",
    linewidth=1,
    label=f"account average ({overall_rate:.3f})",
)
ax.set_xlim(-0.8, 1.6)
ax.set_ylim(-0.45, 0.05)
ax.set_xlabel(r"raw profit per dollar  $r_s = p_s / b_s$")
ax.set_ylabel(r"shrunk profit per dollar  $\hat r_s$")
ax.set_title(
    "Shrinkage pulls noisy ad-group estimates toward the account average\n"
    "(point area proportional to ad-group cost)"
)
ax.legend(loc="lower right")
plt.tight_layout()
save(fig, "fig4_shrinkage_raw_vs_smoothed.png")


# Figure 5: strongest ad groups after smoothing
top10 = ad_group.sort_values("smoothed_profit_per_dollar", ascending=False).head(10).copy()
fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.barh(top10["Ad Group"], top10["smoothed_profit_per_dollar"], color="#4C78A8")
ax.axvline(0, color="black", linewidth=1)
ax.invert_yaxis()
ax.set_xlabel(r"shrunk profit per dollar  $\hat r_s$")
ax.set_title("Top 10 ad groups by shrunk profit-per-dollar (all still negative)")
plt.tight_layout()
save(fig, "fig5_top_ad_group_efficiency.png")


# Figure 6: allocation comparison
fig, ax = plt.subplots(figsize=(8.5, 4.7))
labels = [
    "Current\nobserved",
    "Current\n(smoothed)",
    "Balanced\n($\\alpha$=0.25)",
    "Aggressive\n($\\alpha$=0)",
]
vals = comparison["profit"].values
bars = ax.bar(labels, vals, color=["#E15759", "#F28E2B", "#4C78A8", "#3B5F8A"])
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel(r"expected P&L (\$)")
ax.set_title("Reallocation cuts expected loss; floor controls concentration")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val - 1500,
            (r"\$" + f"{val:,.0f}"),
            ha="center", va="top", fontsize=11)
ax.set_ylim(min(vals) * 1.12, max(vals) + 8000)
plt.tight_layout()
save(fig, "fig6_allocation_comparison.png")


# Figure 7: leave-one-month-out comparison
month_order_short = ["July", "August", "September", "October", "November"]
bt = backtest.set_index("heldout_month").loc[month_order_short].reset_index()
btf = backtest_floor.set_index("heldout_month").loc[month_order_short].reset_index()
fig, ax = plt.subplots(figsize=(9.0, 4.7))
x = np.arange(len(bt))
width = 0.27
ax.bar(x - width, bt["current_profit"], width, label="current allocation", color="#E15759")
ax.bar(x,         btf["model_profit"], width, label=r"balanced ($\alpha$=0.25)", color="#4C78A8")
ax.bar(x + width, bt["model_profit"], width, label=r"aggressive ($\alpha$=0)", color="#3B5F8A")
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(bt["heldout_month"])
ax.set_ylabel(r"held-out month P&L (\$)")
ax.set_title("Leave-one-month-out pseudo-backtest")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
save(fig, "fig7_pseudo_backtest.png")


# Figure 8: cap and smoothing sensitivity
fig, ax = plt.subplots(figsize=(7.0, 4.5))
order = ["weak", "main", "strong"]
colors_smooth = {"weak": "#59A14F", "main": "#4C78A8", "strong": "#F28E2B"}
for label in order:
    g = sensitivity[sensitivity["smoothing"] == label].sort_values("cap_fraction")
    ax.plot(g["cap_fraction"], g["expected_profit"], marker="o",
            label=f"{label} smoothing", color=colors_smooth[label], linewidth=2)
current_observed = comparison.loc[
    comparison["scenario"] == "current observed allocation", "profit"
].iloc[0]
ax.axhline(current_observed, color="black", linestyle="--", linewidth=1.2,
           label=f"current observed ({current_observed:,.0f})")
ax.set_xlabel(r"per-group cap fraction  $q$")
ax.set_ylabel(r"expected P&L (\$)")
ax.set_title("Sensitivity to cap and smoothing strength")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
save(fig, "fig8_allocation_sensitivity.png")


# Figure 9: tiered marginal returns
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(tiered_sens["shrink_rho"], tiered_sens["linear_expected_profit"],
        marker="o", color="#4C78A8", linewidth=2, label="expected P&L (group rates)")
ax.plot(tiered_sens["shrink_rho"], tiered_sens["tiered_objective"],
        marker="o", color="#F28E2B", linewidth=2, label="tiered objective (second tier shrunk)")
main_profit = comparison.loc[
    comparison["scenario"] == "aggressive allocation (alpha=0)", "profit"
].iloc[0]
ax.axhline(main_profit, color="black", linestyle="--", linewidth=1.2,
           label=f"main model ({main_profit:,.0f})")
ax.set_xlabel(r"second-tier weight on group rate  $\rho$")
ax.set_ylabel(r"objective value (\$)")
ax.set_title(r"Tiered marginal-return sensitivity (second tier shrunk toward $r_0$)")
ax.invert_xaxis()
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
save(fig, "fig9_tiered_marginal_sensitivity.png")


# Figure 10: minimum-spend floor
fig, ax = plt.subplots(figsize=(7.5, 4.7))
ax.plot(pareto["floor_fraction"], pareto["expected_profit"],
        marker="o", color="#4C78A8", linewidth=2)
current_observed = comparison.loc[
    comparison["scenario"] == "current observed allocation", "profit"
].iloc[0]
ax.axhline(current_observed, color="#E15759", linestyle="--", linewidth=1.2,
           label=f"current observed ({current_observed:,.0f})")
for _, row in pareto.iterrows():
    floor_fraction = row["floor_fraction"]
    if floor_fraction == pareto["floor_fraction"].min():
        x_offset, alignment = 8, "left"
    elif floor_fraction == pareto["floor_fraction"].max():
        x_offset, alignment = -8, "right"
    else:
        x_offset, alignment = 0, "center"
    ax.annotate(
        f"{int(row['funded_ad_groups'])} funded",
        xy=(floor_fraction, row["expected_profit"]),
        xytext=(x_offset, 12),
        textcoords="offset points",
        ha=alignment,
        fontsize=10,
    )
ax.set_xlabel(r"minimum-spend floor  $\alpha$")
ax.set_ylabel(r"expected P&L (\$)")
ax.set_title("Pareto frontier: aggressiveness vs. operational continuity")
ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.legend(loc="lower left", fontsize=10)
ax.set_ylim(min(pareto["expected_profit"].min(), current_observed) * 1.08,
            pareto["expected_profit"].max() * 0.85)
plt.tight_layout()
save(fig, "fig10_pareto_frontier.png")


print(f"Saved 10 figures to {PLOTS_DIR}")
