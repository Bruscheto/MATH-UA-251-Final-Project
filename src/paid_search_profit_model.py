"""Paid-search budget allocation model for the MATH-UA 251 final project."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final_shop_6modata.csv"
METRICS_DIR = ROOT / "outputs" / "metrics"


# Data preparation

def parse_ad_group(df):
    # Each name contains the match type, device, and keyword theme.
    parts = df["Ad Group"].str.split(" - ", expand=True)
    out = df.copy()
    out["match_type"] = parts[1]
    out["device"] = parts[2]
    out["keyword_theme"] = (
        parts[3]
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
    )
    return out


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return parse_ad_group(df)


# Summary tables

def safe_divide(numerator, denominator):
    """Divide while returning 0 when the denominator is 0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def add_metrics(summary):
    out = summary.copy()
    out["ctr"] = safe_divide(out["clicks"], out["impressions"])
    out["conversion_rate"] = safe_divide(out["conversions"], out["clicks"])
    out["cpc"] = safe_divide(out["cost"], out["clicks"])
    out["cost_per_conversion"] = safe_divide(out["cost"], out["conversions"])
    out["roas"] = safe_divide(out["revenue"], out["cost"])
    out["profit_per_dollar"] = safe_divide(out["profit"], out["cost"])
    return out


def summarize_by(data, group_cols):
    grouped = (
        data.groupby(group_cols, as_index=False)
        .agg(
            rows=("Ad Group", "count"),
            impressions=("Impressions", "sum"),
            clicks=("Clicks", "sum"),
            conversions=("Conversions", "sum"),
            cost=("Cost", "sum"),
            revenue=("Revenue", "sum"),
            sale_amount=("Sale Amount", "sum"),
            profit=("P&L", "sum"),
        )
        .sort_values(group_cols)
    )
    return add_metrics(grouped)


def build_summaries(df):
    monthly = summarize_by(df, ["Month"])
    ad_group = summarize_by(df, ["Ad Group", "match_type", "device", "keyword_theme"])
    segment = summarize_by(df, ["match_type", "device", "keyword_theme"])
    return monthly, ad_group, segment


# Profit-rate estimates

def shrink_table(table, strength=None):
    """Shrink each group's profit rate toward the account-wide rate."""
    out = table.copy()
    total_p = out["profit"].sum()
    total_b = out["cost"].sum()
    r0 = total_p / total_b
    m = out["cost"].median() if strength is None else strength
    out["smoothed_profit_per_dollar"] = (out["profit"] + m * r0) / (out["cost"] + m)
    return out, r0, m


# Budget allocation

def allocate_budget(table, total_budget, cap_fraction,
                    min_spend_floor_fraction=0.0,
                    floor_costs=None,
                    rate_col="smoothed_profit_per_dollar"):
    """Allocate a fixed budget using a cap and an optional spending floor.

        maximize  sum_s x_s * r_hat_s
        subject to  sum_s x_s = B,
                    alpha * b_s  <=  x_s  <=  q * B   for all s.

    The algorithm assigns each group its floor first, then fills the remaining
    budget in descending order of estimated profit per dollar.
    """
    out = table.reset_index(drop=True).copy()
    n = len(out)
    cap = total_budget * cap_fraction

    if floor_costs is None:
        baseline = out["cost"].values.astype(float)
    else:
        baseline = np.asarray(floor_costs, dtype=float)

    floor = np.minimum(min_spend_floor_fraction * baseline, cap)
    floor_total = floor.sum()
    if floor_total > total_budget + 1e-6:
        raise ValueError(
            f"Min-spend floor (${floor_total:,.2f}) exceeds total budget "
            f"(${total_budget:,.2f}). Reduce min_spend_floor_fraction."
        )

    headroom = np.maximum(cap - floor, 0.0)
    allocated = floor.copy()
    remaining = total_budget - floor_total

    rates = out[rate_col].values
    order = np.argsort(-rates)
    for i in order:
        if remaining <= 1e-9:
            break
        take = float(min(headroom[i], remaining))
        allocated[i] += take
        remaining -= take

    out["allocated_budget"] = allocated
    out["expected_profit_allocated"] = allocated * rates
    return out.sort_values("allocated_budget", ascending=False)


def compare_allocations(ad_group, allocation_alpha0, allocation_alpha025):
    total_budget = ad_group["cost"].sum()
    current_profit = ad_group["profit"].sum()
    current_smoothed = (ad_group["cost"] * ad_group["smoothed_profit_per_dollar"]).sum()
    balanced = allocation_alpha025["expected_profit_allocated"].sum()
    aggressive = allocation_alpha0["expected_profit_allocated"].sum()
    return pd.DataFrame(
        [
            {
                "scenario": "current observed allocation",
                "budget": total_budget,
                "profit": current_profit,
            },
            {
                "scenario": "current allocation with smoothed rates",
                "budget": total_budget,
                "profit": current_smoothed,
            },
            {
                "scenario": "balanced allocation (alpha=0.25)",
                "budget": total_budget,
                "profit": balanced,
            },
            {
                "scenario": "aggressive allocation (alpha=0)",
                "budget": total_budget,
                "profit": aggressive,
            },
        ]
    )


def run_pareto_frontier(ad_group, total_budget, cap_fraction=0.25,
                        floor_fractions=(0.0, 0.25, 0.50, 0.75, 1.0)):
    """Compare several values for the minimum-spend floor."""
    rows = []
    for alpha in floor_fractions:
        allocation = allocate_budget(
            ad_group, total_budget, cap_fraction,
            min_spend_floor_fraction=alpha,
        )
        funded = int((allocation["allocated_budget"] > 1.0).sum())
        rows.append({
            "floor_fraction": alpha,
            "funded_ad_groups": funded,
            "expected_profit": allocation["expected_profit_allocated"].sum(),
        })
    return pd.DataFrame(rows)


# Diminishing-returns check

def allocate_budget_tiered(table, total_budget, cap_fraction, shrink_rho, r_account,
                           rate_col="smoothed_profit_per_dollar"):
    """Split each group's cap into two tiers to model diminishing returns.

    The second-tier rate is pulled toward the account average by ``shrink_rho``.
    """
    cap = total_budget * cap_fraction
    half = cap / 2.0

    rows = table.reset_index(drop=True)
    n = len(rows)
    allocated = np.zeros(n)
    tiered_objective = np.zeros(n)

    buckets = []
    for i, row in rows.iterrows():
        base_rate = row[rate_col]
        second_rate = shrink_rho * base_rate + (1.0 - shrink_rho) * r_account
        second_rate = min(second_rate, base_rate)
        buckets.append((i, half, base_rate))
        buckets.append((i, half, second_rate))
    buckets.sort(key=lambda b: b[2], reverse=True)

    remaining = total_budget
    for i, capacity, rate in buckets:
        if remaining <= 1e-9:
            break
        amount = min(capacity, remaining)
        allocated[i] += amount
        tiered_objective[i] += amount * rate
        remaining -= amount

    out = rows.copy()
    out["allocated_budget"] = allocated
    out["expected_profit_allocated"] = allocated * out[rate_col]
    out["tier_shrink_rho"] = shrink_rho
    out["tiered_objective"] = tiered_objective
    return out.sort_values("allocated_budget", ascending=False)


def run_tiered_sensitivity(ad_group, total_budget, r_account, cap_fraction=0.25):
    rows = []
    main = None
    for shrink_rho in [1.0, 0.75, 0.5, 0.25, 0.0]:
        allocation = allocate_budget_tiered(
            ad_group, total_budget, cap_fraction, shrink_rho, r_account
        )
        rows.append({
            "shrink_rho": shrink_rho,
            "allocated_ad_groups": int((allocation["allocated_budget"] > 1e-6).sum()),
            "linear_expected_profit": allocation["expected_profit_allocated"].sum(),
            "tiered_objective": allocation["tiered_objective"].sum(),
        })
        if shrink_rho == 0.5:
            main = allocation
    return pd.DataFrame(rows), main


# Leave-one-month-out check

def run_pseudo_backtest(df, cap_fraction=0.25, min_spend_floor_fraction=0.0):
    """Train on four months and score the allocation on the month left out."""
    rows = []
    months = sorted(df["Month"].unique())
    for heldout_month in months:
        train = df[df["Month"] != heldout_month]
        test = df[df["Month"] == heldout_month]

        group_cols = ["Ad Group", "match_type", "device", "keyword_theme"]
        train_ad = summarize_by(train, group_cols)
        test_ad = summarize_by(test, group_cols)
        train_ad, _, _ = shrink_table(train_ad)

        common = sorted(set(train_ad["Ad Group"]).intersection(test_ad["Ad Group"]))
        train_eval = train_ad[train_ad["Ad Group"].isin(common)].copy()
        test_eval = test_ad[test_ad["Ad Group"].isin(common)].copy()
        heldout_budget = test_eval["cost"].sum()

        train_total = float(train_eval["cost"].sum())
        if train_total > 0:
            floor_baseline = train_eval["cost"].values * (heldout_budget / train_total)
        else:
            floor_baseline = train_eval["cost"].values

        allocation = allocate_budget(
            train_eval,
            total_budget=heldout_budget,
            cap_fraction=cap_fraction,
            min_spend_floor_fraction=min_spend_floor_fraction,
            floor_costs=floor_baseline,
        )
        rates = test_eval[["Ad Group", "profit_per_dollar"]].rename(
            columns={"profit_per_dollar": "heldout_rate"}
        )
        evaluated = allocation.merge(rates, on="Ad Group", how="left")
        model_profit = (evaluated["allocated_budget"] * evaluated["heldout_rate"]).sum()
        current_profit = test_eval["profit"].sum()

        rows.append({
            "heldout_month": heldout_month,
            "evaluable_ad_groups": len(common),
            "heldout_budget": heldout_budget,
            "current_profit": current_profit,
            "model_profit": model_profit,
            "improvement": model_profit - current_profit,
            "floor_fraction": min_spend_floor_fraction,
        })
    return pd.DataFrame(rows)


# Parameter sensitivity

def run_sensitivity(ad_group, smoothing_strength, overall_rate):
    """Re-run the model for three smoothing strengths and three caps."""
    rows = []
    total_budget = ad_group["cost"].sum()
    base = ad_group.drop(columns=["smoothed_profit_per_dollar"], errors="ignore")

    main_smoothed, _, _ = shrink_table(base, strength=smoothing_strength)
    main_alloc = allocate_budget(main_smoothed, total_budget=total_budget, cap_fraction=0.25)
    main_top4 = set(main_alloc.nlargest(4, "allocated_budget")["Ad Group"])

    smoothers = {
        "weak": smoothing_strength * 0.5,
        "main": smoothing_strength,
        "strong": smoothing_strength * 2.0,
    }
    for label, strength in smoothers.items():
        smoothed, _, _ = shrink_table(base, strength=strength)
        for cap in [0.15, 0.25, 0.40]:
            allocation = allocate_budget(
                smoothed, total_budget=total_budget, cap_fraction=cap
            )
            top4 = set(allocation.nlargest(4, "allocated_budget")["Ad Group"])
            top4_intersect = len(top4 & main_top4)
            rows.append({
                "smoothing": label,
                "smoothing_strength": strength,
                "cap_fraction": cap,
                "overall_profit_per_dollar": overall_rate,
                "expected_profit": allocation["expected_profit_allocated"].sum(),
                "top4_overlap_with_main": top4_intersect,
            })
    return pd.DataFrame(rows)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    monthly, ad_group, segment = build_summaries(df)

    ad_group, overall_rate, smoothing_strength = shrink_table(ad_group)
    total_budget = ad_group["cost"].sum()

    allocation_balanced = allocate_budget(
        ad_group, total_budget=total_budget, cap_fraction=0.25,
        min_spend_floor_fraction=0.25,
    )
    allocation_aggressive = allocate_budget(
        ad_group, total_budget=total_budget, cap_fraction=0.25,
        min_spend_floor_fraction=0.0,
    )
    comparison = compare_allocations(ad_group, allocation_aggressive, allocation_balanced)

    pareto = run_pareto_frontier(ad_group, total_budget, cap_fraction=0.25)

    tiered_sensitivity, tiered_main = run_tiered_sensitivity(
        ad_group, total_budget, r_account=overall_rate
    )

    backtest_alpha0 = run_pseudo_backtest(df, min_spend_floor_fraction=0.0)
    backtest_alpha025 = run_pseudo_backtest(df, min_spend_floor_fraction=0.25)

    sensitivity = run_sensitivity(ad_group, smoothing_strength, overall_rate)

    monthly.to_csv(METRICS_DIR / "monthly_summary.csv", index=False)
    ad_group.to_csv(METRICS_DIR / "ad_group_summary.csv", index=False)
    segment.to_csv(METRICS_DIR / "segment_summary.csv", index=False)
    allocation_balanced.to_csv(METRICS_DIR / "allocation_main.csv", index=False)
    allocation_aggressive.to_csv(METRICS_DIR / "allocation_aggressive.csv", index=False)
    comparison.to_csv(METRICS_DIR / "allocation_comparison.csv", index=False)
    pareto.to_csv(METRICS_DIR / "pareto_frontier.csv", index=False)
    sensitivity.to_csv(METRICS_DIR / "allocation_sensitivity.csv", index=False)
    tiered_sensitivity.to_csv(METRICS_DIR / "tiered_marginal_sensitivity.csv", index=False)
    tiered_main.to_csv(METRICS_DIR / "tiered_marginal_allocation_rho_050.csv", index=False)
    backtest_alpha0.to_csv(METRICS_DIR / "pseudo_backtest.csv", index=False)
    backtest_alpha025.to_csv(METRICS_DIR / "pseudo_backtest_floor.csv", index=False)

    print("Saved summary tables.")
    print(f"overall profit per dollar: {overall_rate:.4f}")
    print(f"smoothing strength (median ad-group cost): ${smoothing_strength:,.0f}")
    print(f"balanced allocation (alpha=0.25) expected profit: "
          f"${allocation_balanced['expected_profit_allocated'].sum():,.2f}")
    print(f"aggressive allocation (alpha=0) expected profit:  "
          f"${allocation_aggressive['expected_profit_allocated'].sum():,.2f}")
    print(f"tiered allocation (rho=0.5) expected profit:      "
          f"${tiered_main['expected_profit_allocated'].sum():,.2f}")
    print(f"backtest improvement over current (alpha=0):      "
          f"${backtest_alpha0['improvement'].sum():,.2f}")
    print(f"backtest improvement over current (alpha=0.25):   "
          f"${backtest_alpha025['improvement'].sum():,.2f}")


if __name__ == "__main__":
    main()
