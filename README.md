# Entropic-Dynamics-of-the-Universal-Equivalent
"""
Section 6: Agent-Based Numerical Simulation of Endogenous Crisis Formation
Stable Colab Version

Three agent classes:
1. Liquidity Providers
2. Leveraged Speculators
3. Crisis-Sensitive Reallocators

Outputs:
- ABM time series CSV
- Summary statistics CSV
- 7 manuscript-ready figures
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy


# ============================================================
# 0. CONFIGURATION
# ============================================================

class ABMConfig:
    seed = 42
    n_steps = 30000
    dt_minutes = 1

    output_dir = "output_abm_stable"

    initial_price = 1500.0
    initial_fundamental = 1500.0

    # Agent populations
    n_lp = 50
    n_spec = 100
    n_csr = 60

    # Market structure
    base_depth = 8000.0
    min_depth = 800.0
    base_spread = 0.00012
    impact_scale = 0.000035

    # Fundamental dynamics
    fundamental_sigma = 0.000025
    mean_reversion = 0.030

    # Liquidity providers
    lp_inventory_limit = 150.0
    lp_withdraw_slope = 7.0
    lp_inventory_spread_weight = 0.00012
    lp_vol_spread_weight = 8.0

    # Speculators
    spec_noise_weight = 3.0
    spec_trend_weight = 5500.0
    spec_leverage_mean = 1.5
    spec_leverage_std = 0.25
    forced_deleveraging_weight = 35.0
    margin_stress_threshold = 0.68

    # Crisis-sensitive reallocators
    csr_stress_threshold = 0.46
    csr_slope = 12.0
    csr_order_scale = 22.0

    # Stress equation
    stress_decay = 0.985
    stress_return_weight = 30.0
    stress_spread_weight = 22.0
    stress_imbalance_weight = 0.20
    stress_depth_weight = 0.30

    # Numerical safety
    max_abs_return = 0.025

    # Rolling diagnostics
    metric_window = 500
    entropy_bins = 35
    hill_tail_fraction = 0.08


CFG = ABMConfig()
os.makedirs(CFG.output_dir, exist_ok=True)


# ============================================================
# 1. UTILITY FUNCTIONS
# ============================================================

def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def zscore_series(s):
    s = pd.Series(s)
    std = s.std(skipna=True)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / std


def calc_shannon_entropy(data, bins=35):
    x = np.asarray(data)
    x = x[np.isfinite(x)]

    if len(x) < 30 or np.std(x) == 0:
        return np.nan

    counts, _ = np.histogram(x, bins=bins)
    probs = counts[counts > 0].astype(float)
    probs /= probs.sum()

    return float(scipy_entropy(probs))


def tsallis_entropy(data, q=1.5, bins=35):
    x = np.asarray(data)
    x = x[np.isfinite(x)]

    if len(x) < 30 or np.std(x) == 0:
        return np.nan

    counts, _ = np.histogram(x, bins=bins)
    probs = counts[counts > 0].astype(float)
    probs /= probs.sum()

    return float((1.0 - np.sum(probs ** q)) / (q - 1.0))


def permutation_entropy(data, m=3, tau=1):
    x = np.asarray(data)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < m * tau + 10:
        return np.nan

    patterns = {}

    for i in range(n - tau * (m - 1)):
        window = x[i:i + tau * m:tau]
        pattern = tuple(np.argsort(window))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()

    h = -np.sum(probs * np.log(probs))
    h_max = np.log(math.factorial(m))

    return float(h / h_max)


def hill_estimator_losses(losses, tail_fraction=0.08):
    x = np.asarray(losses)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    if len(x) < 40:
        return np.nan

    x_sorted = np.sort(x)[::-1]
    k = max(8, int(len(x_sorted) * tail_fraction))

    if k + 1 >= len(x_sorted):
        return np.nan

    threshold = x_sorted[k]

    if threshold <= 0:
        return np.nan

    return float(np.mean(np.log(x_sorted[:k] / threshold)))


# ============================================================
# 2. AGENT-BASED MODEL
# ============================================================

def run_abm(cfg):
    rng = np.random.default_rng(cfg.seed)

    time_index = pd.date_range(
        start="2020-01-01",
        periods=cfg.n_steps,
        freq=f"{cfg.dt_minutes}min"
    )

    log_price = np.zeros(cfg.n_steps)
    log_fundamental = np.zeros(cfg.n_steps)

    log_price[0] = np.log(cfg.initial_price)
    log_fundamental[0] = np.log(cfg.initial_fundamental)

    returns = np.zeros(cfg.n_steps)
    spread = np.zeros(cfg.n_steps)
    depth = np.zeros(cfg.n_steps)
    stress = np.zeros(cfg.n_steps)
    imbalance = np.zeros(cfg.n_steps)

    lp_flow = np.zeros(cfg.n_steps)
    spec_flow = np.zeros(cfg.n_steps)
    csr_flow = np.zeros(cfg.n_steps)
    forced_flow = np.zeros(cfg.n_steps)
    lp_activity = np.zeros(cfg.n_steps)

    spread[0] = cfg.base_spread
    depth[0] = cfg.base_depth
    stress[0] = 0.03
    lp_activity[0] = 1.0

    # Agent states
    lp_inventory = rng.normal(0, 4, cfg.n_lp)

    spec_positions = rng.normal(0, 1, cfg.n_spec)
    spec_leverage = np.clip(
        rng.normal(cfg.spec_leverage_mean, cfg.spec_leverage_std, cfg.n_spec),
        0.7,
        2.5
    )
    spec_sensitivity = np.clip(
        rng.normal(1.0, 0.25, cfg.n_spec),
        0.4,
        1.8
    )

    csr_thresholds = np.clip(
        rng.normal(cfg.csr_stress_threshold, 0.07, cfg.n_csr),
        0.25,
        0.75
    )
    csr_intensity = np.clip(
        rng.normal(1.0, 0.25, cfg.n_csr),
        0.5,
        1.6
    )

    for t in range(1, cfg.n_steps):

        # ----------------------------
        # Fundamental value
        # ----------------------------
        log_fundamental[t] = (
            log_fundamental[t - 1]
            + rng.normal(0.0, cfg.fundamental_sigma)
        )

        mispricing = log_price[t - 1] - log_fundamental[t]

        # ----------------------------
        # Local volatility and trend
        # ----------------------------
        if t >= 30:
            recent = returns[t - 30:t]
            short_vol = np.std(recent)
            trend = np.mean(recent)
        else:
            short_vol = abs(returns[t - 1])
            trend = returns[t - 1]

        prev_stress = stress[t - 1]

        # ----------------------------
        # Liquidity providers
        # ----------------------------
        active_prob = 1.0 - sigmoid(
            cfg.lp_withdraw_slope * (prev_stress - 0.55)
        )

        active_prob = np.clip(active_prob, 0.05, 1.0)

        lp_orders = (
            -15.0 * mispricing
            -0.08 * lp_inventory
            + rng.normal(0, 1.5, cfg.n_lp)
        )

        lp_orders *= active_prob

        lp_net = lp_orders.sum()

        lp_inventory += 0.01 * lp_orders
        lp_inventory = np.clip(
            lp_inventory,
            -cfg.lp_inventory_limit,
            cfg.lp_inventory_limit
        )

        # ----------------------------
        # Leveraged speculators
        # ----------------------------
        spec_noise = rng.normal(0, cfg.spec_noise_weight, cfg.n_spec)

        spec_trend_orders = (
            cfg.spec_trend_weight
            * trend
            * spec_leverage
            * spec_sensitivity
        )

        spec_orders = spec_trend_orders + spec_noise

        margin_prob = sigmoid(
            14.0 * (prev_stress - cfg.margin_stress_threshold)
        )

        margin_flags = rng.uniform(size=cfg.n_spec) < margin_prob

        deleveraging = np.zeros(cfg.n_spec)
        deleveraging[margin_flags] = (
            -cfg.forced_deleveraging_weight
            * np.sign(spec_positions[margin_flags] + 1e-8)
        )

        spec_orders += deleveraging

        spec_positions += 0.02 * spec_orders
        spec_positions = np.clip(spec_positions, -80, 80)

        spec_net = spec_orders.sum()
        forced_net = deleveraging.sum()

        # ----------------------------
        # Crisis-sensitive reallocators
        # ----------------------------
        csr_activation = sigmoid(
            cfg.csr_slope * (prev_stress - csr_thresholds)
        )

        csr_orders = (
            cfg.csr_order_scale
            * csr_intensity
            * csr_activation
            + rng.normal(0, 1.2, cfg.n_csr)
        )

        csr_net = csr_orders.sum()

        # ----------------------------
        # Aggregate flow
        # ----------------------------
        total_order = lp_net + spec_net + csr_net

        gross_flow = (
            abs(lp_net)
            + abs(spec_net)
            + abs(csr_net)
            + 1e-8
        )

        imb = total_order / gross_flow

        # ----------------------------
        # Endogenous depth
        # ----------------------------
        depth_t = (
            cfg.base_depth
            * (1.0 - 0.55 * min(prev_stress, 1.0))
            * np.exp(-6.0 * short_vol)
            * (0.65 + 0.35 * active_prob)
        )

        depth_t = np.clip(depth_t, cfg.min_depth, cfg.base_depth)

        # ----------------------------
        # Endogenous spread
        # ----------------------------
        inv_pressure = np.mean(np.abs(lp_inventory)) / cfg.lp_inventory_limit

        spread_t = (
            cfg.base_spread
            + cfg.lp_vol_spread_weight * short_vol
            + cfg.lp_inventory_spread_weight * inv_pressure
            + 0.0015 * (1.0 - active_prob)
        )

        spread_t = np.clip(spread_t, cfg.base_spread, 0.025)

        # ----------------------------
        # Price impact
        # ----------------------------
        impact = cfg.impact_scale * total_order / depth_t
        reversion = -cfg.mean_reversion * mispricing
        noise = rng.normal(0, 0.00008 + 0.15 * short_vol)

        jump_prob = sigmoid(10.0 * (prev_stress - 0.75)) * min(1.0, 2.5 * abs(imb))

        jump = 0.0
        if rng.uniform() < jump_prob:
            jump = rng.normal(0, 0.0025 + 1.5 * short_vol)

        r_t = reversion + impact + noise + jump

        # numerical safety
        r_t = np.clip(r_t, -cfg.max_abs_return, cfg.max_abs_return)

        returns[t] = r_t
        log_price[t] = log_price[t - 1] + r_t

        # ----------------------------
        # Stress update
        # ----------------------------
        stress_innovation = (
            cfg.stress_return_weight * abs(r_t)
            + cfg.stress_spread_weight * spread_t
            + cfg.stress_imbalance_weight * abs(imb)
            + cfg.stress_depth_weight * (1.0 - depth_t / cfg.base_depth)
        )

        stress[t] = np.clip(
            cfg.stress_decay * prev_stress
            + (1.0 - cfg.stress_decay) * stress_innovation,
            0.0,
            1.0
        )

        spread[t] = spread_t
        depth[t] = depth_t
        imbalance[t] = imb
        lp_flow[t] = lp_net
        spec_flow[t] = spec_net
        csr_flow[t] = csr_net
        forced_flow[t] = forced_net
        lp_activity[t] = active_prob

    df = pd.DataFrame(
        {
            "price": np.exp(log_price),
            "fundamental": np.exp(log_fundamental),
            "log_price": log_price,
            "log_fundamental": log_fundamental,
            "return": returns,
            "spread": spread,
            "depth": depth,
            "stress": stress,
            "imbalance": imbalance,
            "lp_flow": lp_flow,
            "spec_flow": spec_flow,
            "csr_flow": csr_flow,
            "forced_deleveraging_flow": forced_flow,
            "lp_activity": lp_activity,
        },
        index=time_index
    )

    return df


# ============================================================
# 3. ROLLING ENTROPY, TAIL, LIQUIDITY, GAMMA
# ============================================================

def compute_metrics(df, cfg):
    out = df.copy()
    w = cfg.metric_window

    out["H_shannon"] = (
        out["return"]
        .rolling(w)
        .apply(lambda z: calc_shannon_entropy(z, bins=cfg.entropy_bins), raw=True)
    )

    out["H_tsallis"] = (
        out["return"]
        .rolling(w)
        .apply(lambda z: tsallis_entropy(z, q=1.5, bins=cfg.entropy_bins), raw=True)
    )

    out["H_perm"] = (
        out["return"]
        .rolling(w)
        .apply(lambda z: permutation_entropy(z, m=3, tau=1), raw=True)
    )

    losses = -out["return"]

    out["xi_hill"] = (
        losses
        .rolling(w)
        .apply(
            lambda z: hill_estimator_losses(
                z,
                tail_fraction=cfg.hill_tail_fraction
            ),
            raw=True
        )
    )

    out["rv"] = out["return"].rolling(w).std()

    L_components = pd.DataFrame(index=out.index)
    L_components["spread_z"] = zscore_series(out["spread"])
    L_components["rv_z"] = zscore_series(out["rv"])
    L_components["imbalance_z"] = zscore_series(np.abs(out["imbalance"]))
    L_components["inv_depth_z"] = zscore_series(1.0 / out["depth"])

    out["L_t_raw"] = L_components.mean(axis=1)

    # Positive liquidity stress
    out["L_t"] = out["L_t_raw"] - out["L_t_raw"].min(skipna=True) + 1e-6

    # Effective market depth proxy
    out["Lambda_t"] = 1.0 / (1.0 + out["L_t"])

    # Entropy-tail instability metric
    out["Gamma_t"] = (
        out["H_tsallis"]
        * out["xi_hill"]
        * (1.0 + out["L_t"])
    )

    return out


# ============================================================
# 4. SUMMARY TABLE
# ============================================================

def build_summary(df, cfg):
    valid = df.dropna(subset=["H_tsallis", "xi_hill", "Gamma_t"])

    calm = valid[valid["stress"] <= valid["stress"].quantile(0.30)]
    crisis = valid[valid["stress"] >= valid["stress"].quantile(0.80)]

    rows = []

    for name, d in [("Calm Regime", calm), ("Crisis Regime", crisis)]:
        rows.append(
            {
                "Regime": name,
                "Mean Return": d["return"].mean(),
                "Std Return": d["return"].std(),
                "Mean Spread": d["spread"].mean(),
                "Mean Depth": d["depth"].mean(),
                "Mean Stress": d["stress"].mean(),
                "Mean Shannon": d["H_shannon"].mean(),
                "Mean Tsallis": d["H_tsallis"].mean(),
                "Mean Permutation": d["H_perm"].mean(),
                "Mean Hill Xi": d["xi_hill"].mean(),
                "Mean Gamma": d["Gamma_t"].mean(),
                "Max Gamma": d["Gamma_t"].max(),
            }
        )

    summary = pd.DataFrame(rows)
    return summary


# ============================================================
# 5. FIGURES
# ============================================================

def crisis_time(df):
    idx = df.index[df["stress"] > 0.65]
    if len(idx) > 0:
        return idx[0]
    return df.index[int(len(df) * 0.65)]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


def make_figures(df, cfg):
    ctime = crisis_time(df)

    # Figure 1
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["price"], label="Market Price", linewidth=1.2)
    plt.plot(df.index, df["fundamental"], label="Fundamental", linewidth=1.0)
    plt.axvline(ctime, linestyle="--", label="Crisis Threshold")
    plt.title("Figure 1: ABM Price Dynamics")
    plt.ylabel("Price")
    plt.legend()
    savefig(os.path.join(cfg.output_dir, "Fig1_ABM_Price.png"))

    # Figure 2
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["H_tsallis"], label="Tsallis Entropy", linewidth=1.2)
    plt.plot(df.index, df["H_shannon"], label="Shannon Entropy", linewidth=1.0)
    plt.axvline(ctime, linestyle="--", label="Crisis Threshold")
    plt.title("Figure 2: Entropy Escalation")
    plt.ylabel("Entropy")
    plt.legend()
    savefig(os.path.join(cfg.output_dir, "Fig2_ABM_Entropy.png"))

    # Figure 3
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["xi_hill"], linewidth=1.2)
    plt.axvline(ctime, linestyle="--", label="Crisis Threshold")
    plt.title("Figure 3: Tail-Risk Amplification")
    plt.ylabel("Hill Tail Index")
    plt.legend()
    savefig(os.path.join(cfg.output_dir, "Fig3_ABM_TailRisk.png"))

    # Figure 4
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["L_t"], label="Liquidity Stress $L_t$", linewidth=1.2)
    plt.plot(df.index, df["spread"] * 10000, label="Spread, bp", linewidth=1.0)
    plt.axvline(ctime, linestyle="--", label="Crisis Threshold")
    plt.title("Figure 4: Liquidity Deterioration")
    plt.ylabel("Stress / Basis Points")
    plt.legend()
    savefig(os.path.join(cfg.output_dir, "Fig4_ABM_Liquidity.png"))

    # Figure 5
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Gamma_t"], linewidth=1.3)
    plt.axvline(ctime, linestyle="--", label="Crisis Threshold")
    plt.title(r"Figure 5: Entropy–Tail Instability Metric $\Gamma_t$")
    plt.ylabel(r"$\Gamma_t$")
    plt.legend()
    savefig(os.path.join(cfg.output_dir, "Fig5_ABM_Gamma.png"))

    # Figure 6
    valid = df[["H_tsallis", "xi_hill", "spread", "stress"]].dropna()

    plt.figure(figsize=(8.5, 6.5))

    calm = valid["stress"] <= valid["stress"].quantile(0.40)
    crisis = valid["stress"] >= valid["stress"].quantile(0.80)

    plt.scatter(
        valid.loc[calm, "H_tsallis"],
        valid.loc[calm, "xi_hill"],
        s=12,
        alpha=0.25,
        label="Calm Regime"
    )

    sc = plt.scatter(
        valid.loc[crisis, "H_tsallis"],
        valid.loc[crisis, "xi_hill"],
        c=valid.loc[crisis, "spread"],
        cmap="magma",
        s=14,
        alpha=0.65,
        label="Crisis Regime"
    )

    plt.colorbar(sc, label="Spread")
    plt.xlabel("Tsallis Entropy")
    plt.ylabel("Hill Tail Index")
    plt.title("Figure 6: Non-Equilibrium Phase Transition")
    plt.legend()
    savefig(os.path.join(cfg.output_dir, "Fig6_ABM_PhaseSpace.png"))

    # Figure 7
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["lp_flow"], label="Liquidity Providers", linewidth=0.8)
    plt.plot(df.index, df["spec_flow"], label="Leveraged Speculators", linewidth=0.8)
    plt.plot(df.index, df["csr_flow"], label="Crisis-Sensitive Reallocators", linewidth=0.8)
    plt.plot(df.index, df["forced_deleveraging_flow"], label="Forced Deleveraging", linewidth=0.8)
    plt.axvline(ctime, linestyle="--", label="Crisis Threshold")
    plt.title("Figure 7: Agent Flow Reconfiguration")
    plt.ylabel("Net Order Flow")
    plt.legend(ncol=2)
    savefig(os.path.join(cfg.output_dir, "Fig7_ABM_AgentFlows.png"))


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("Running stable three-class ABM...")

    df = run_abm(CFG)

    print("Computing rolling entropy, tail-risk, liquidity stress, and Gamma...")

    df = compute_metrics(df, CFG)

    print("Building summary statistics...")

    summary = build_summary(df, CFG)

    print(summary)

    df.to_csv(os.path.join(CFG.output_dir, "abm_timeseries_stable.csv"))
    summary.to_csv(os.path.join(CFG.output_dir, "abm_summary_stable.csv"), index=False)

    print("Generating figures...")

    make_figures(df, CFG)

    print(f"Done. All outputs saved in: {CFG.output_dir}/")

    return df, summary


df, summary = main()
