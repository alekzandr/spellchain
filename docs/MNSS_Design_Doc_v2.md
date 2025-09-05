# MNSS (Meta-Normalized Synergy Score) — Design Document v2
**Version:** 2.0  
**Date:** 2025-09-03  
**Scope:** Standard (60-card) decks.  
**Goal:** Provide a consistent, statistically grounded score for comparing candidate Standard decks to a fixed Baseline Meta, emphasizing plan coherence and matchup-relevant capabilities.

---

## 1) Overview

MNSS maps several raw deck quality metrics onto a common 0–1 scale using z-scores against a **Baseline Meta** (a set of top meta decks). Those normalized metrics are then combined (weighted sum) and expressed on a 0–100 scale.

**MNSS = 100 × [0.25·PC + 0.20·ES + 0.10·CS + 0.15·MR + 0.10·RD + 0.10·VEL + 0.10·IF]**

- **PC** — Primary Cluster density (how “on-plan” the maindeck is)  
- **ES** — Engine Support (enablers/payoffs density for the deck’s plan)  
- **CS** — Curve Smoothness (lower MV spread → smoother deployment)  
- **MR** — Mana Reliability (weighted color sources vs. colored pip demand)  
- **RD** — Functional Redundancy (depth in plan-critical roles + healthy breadth)  
- **VEL** — Velocity (cheap smoothing: cantrips, loot, scry, etc.)  
- **IF** — Interaction Fit (ratio of cheap interaction among total interaction)

All per-metric raw values are computed for the evaluated deck, converted into z-scores using the Baseline Meta means and standard deviations, then mapped to [0,1] and combined with the weights above.

---

## 2) Baseline Meta

- The Baseline Meta is a frozen set of representative top decks. For each metric **m**, compute and store the mean **μ_m** and standard deviation **σ_m** across the Baseline Meta.
- Recompute Baseline parameters only when updating the Baseline Meta snapshot. Store these alongside the versioned Baseline for reproducibility.

---

## 3) Normalization Function (single, consistent)

We use a single linear map from z-score to [0,1].

- **z-score:** `z_m = (raw_m − μ_m) / max(σ_m, 1e−6)`  
- **Mapper:** `map_z_to_unit(z) = clip(0.5 + 0.2·z, 0, 1)`  
  - Center (z=0) → 0.5; ±2.5σ map to 0 and 1 respectively.
  - Use **this** named function everywhere (not “norm(x)”). It expects **z**.

> Alternative (optional): replace `map_z_to_unit(z)` with the Normal CDF `Φ(z)` if smoother tails are desired. Use exactly one choice throughout for consistency.

---

## 4) Units & Common Denominators

Unless otherwise stated, all density-style metrics divide by **#nonland** (60 − lands):
- `#nonland = max(1, 60 − #lands)`  
- “Density” = **hits / #nonland**.

This ensures comparisons across different land counts remain stable.

---

## 5) Metric Definitions

### 5.1 Primary Cluster (PC)
Identify the deck’s **Primary Cluster** (game plan) via lexicon matches (payoffs/enablers). If multiple clusters tie, use the tie-break in §7.

- `PC_raw = primary_cluster_copies / #nonland`  
- `PC = map_z_to_unit( z(PC_raw) )`

### 5.2 Engine Support (ES)
Express each plan’s support as **densities**—not arbitrary caps. The Baseline normalization handles scale automatically.

- **Spellchain:** `ES_raw = 0.5·cheap_interaction_ratio + 0.5·velocity_density`  
- **Landfall:**  `ES_raw = 0.6·fetch_like_enabler_density + 0.4·landfall_payoff_density`  
- **Tokens:**    `ES_raw = 0.5·(payoff_density / max(1e−6, token_maker_density)) + 0.5·token_maker_density`  
- **Control:**   `ES_raw = 0.5·cheap_interaction_ratio + 0.5·min(1, interaction_density)`  
- **Graveyard:** `ES_raw = 0.6·gy_enabler_density + 0.4·removal_density`  

Then `ES = map_z_to_unit( z(ES_raw) )`.

> Notes:  
> - `cheap_interaction_ratio = cheap_interaction_hits / max(1, total_interaction_hits)`  
> - Each `*_density = hits / #nonland` (see §4).  
> - Use reasonable lexicons per cluster (kept external to this doc for maintainability).

### 5.3 Curve Smoothness (CS)
Lower spread of nonland MV indicates smoother deployment.

- `std_cmc = stdev(MV of all nonland cards, counting copies)`  
- `z_std = z(std_cmc)`; lower is better, so **invert the sign**: `z_CS = −z_std`  
- `CS = map_z_to_unit(z_CS)`

This avoids mixed scaling and “double normalization.”

### 5.4 Mana Reliability (MR)
**Objective:** Compare **weighted color sources** to **color demand** (pips), with early-turn realism and guardrails.

#### 5.4.1 Weighted Sources per color *c*
Assign weights per land/card producing color *c*:
- Untapped, unconditional source: **1.0**
- ETB tapped (gainlands, tri-lands, etc.): **0.9**
- Conditionals (“two or more lands”, check-lands for early turns): **0.8**
- MDFC where color exists only on one face: **0.5** (unless deck forces that face; then 1.0)
- Treasure/Blood/etc.: **do not** count as fixed sources for MR (they can influence ES).

Sum these to get `sources_c_weighted`.

#### 5.4.2 Targets per color *c*
Demand depends on early drops and pip intensity:
- If color *c* has **any 1-drop** or **≥ 8 double-pip spells**, set `target_c = 14`
- Else if color *c* has **any double pips**, set `target_c = 12`
- Else (splash/single pips only), `target_c = 8`

#### 5.4.3 Pip Share
Let `pips_c = total number of colored pips of color c across all nonland spells (counting copies)`.  
`pip_share_c = pips_c / max(1, Σ over colors pips_color)`

#### 5.4.4 Combine
- `per_color = min(1.2, sources_c_weighted / target_c)`  
- `MR_raw = Σ_c per_color · pip_share_c`  
- `MR = map_z_to_unit( z(MR_raw) )`

### 5.5 Functional Redundancy (RD)
Reward **depth** in plan-critical roles first, then modest breadth.

- Define **plan-critical roles** = {{removal, countermagic, payoff, velocity}}.  
- `depth_count = #roles with depth ≥ 4 among plan-critical roles` (cap at 4).  
- `breadth = min(1, distinct_roles / 6)` (distinct_roles includes utility roles).  
- `RD_raw = 0.7·(depth_count / 4) + 0.3·breadth`  
- `RD = map_z_to_unit( z(RD_raw) )`

### 5.6 Velocity (VEL)
Measure **smoothing**, not just raw card draw volume.

- Count a **VEL hit** if:
  - Spell has **MV ≤ 2** and text includes “draw a card”, **or**
  - Any MV with **scry**, **loot**, **connive**, **rummage**, **surveil** (selection mechanics).
- `VEL_raw = VEL_hits / #nonland`  
- `VEL = map_z_to_unit( z(VEL_raw) )`

### 5.7 Interaction Fit (IF)
Ensure cheap interaction is a healthy fraction of total interaction. Include zero-division guard in spec.

- `IF_raw = cheap_interaction_hits / max(1, total_interaction_hits)`  
- `IF = map_z_to_unit( z(IF_raw) )`

---

## 6) Aggregation

The final score is a weighted sum of normalized metrics:

```
MNSS = 100 × [0.25·PC + 0.20·ES + 0.10·CS + 0.15·MR + 0.10·RD + 0.10·VEL + 0.10·IF]
```

Weights sum to 1.0. Changing weights requires re-validating against the Baseline Meta to maintain calibration.

---

## 7) Cluster Identification & Tie-breaks

1) Identify the candidate clusters by counting **payoff_hits** and **enabler_hits** per cluster via lexicons.  
2) Primary cluster = the cluster with the highest `(payoff_hits + enabler_hits)` **density**.  
3) **Tie-break rule:** Choose the cluster with the higher `(payoff_hits − enabler_hits)`.  
4) If still tied, priority order: **Spellchain > Landfall > Tokens > Control > Graveyard**.

---

## 8) Edge Cases & Rulings

- **Tri-color & greedy manabases:** Use the weighted-source table (untapped vs tapped vs conditional) rather than counting all sources equally. MR catches over-greed.  
- **MDFC & Adventure cards:** Count color for MR per the face actually producing mana (MDFC) or colored pips on the spell portion (Adventure). MDFC off-color face counts **0.5** unless the deck strategy forces that face reliably.  
- **Treasure/Blood/Powerstones:** Do **not** count as fixed color sources for MR. They may influence ES via enabler densities.  
- **Per-metric units:** When “density” is used, divide by `#nonland`. PC already divides by `#nonland`.  
- **Zero divisions:** Always use `max(1, …)` for denominators counting cards.  
- **Capping:** Only use explicit caps where counts can explode (e.g., `min(1, interaction_density)` in Control’s ES). Let Baseline normalization do most scaling.

---

## 9) Pseudocode (reference)

```python
# Inputs:
# deck: list of (card_name, count, mv, colors, types, text, tags)
# baseline: dict with mu/sigma per metric m in {{PC, ES, CS, MR, RD, VEL, IF}}

def map_z_to_unit(z: float) -> float:
    return max(0.0, min(1.0, 0.5 + 0.2 * z))

def zscore(raw: float, mu: float, sigma: float) -> float:
    return (raw - mu) / max(sigma, 1e-6)

def nonland_count(deck):
    return max(1, sum(cnt for _, cnt, *rest in deck if "Land" not in rest[3]))

def density(hits, nonlands):
    return hits / max(1, nonlands)

# --- Metrics ---
def PC_raw(deck):
    # determine primary cluster via lexicons
    cluster_stats = compute_cluster_stats(deck)  # returns per-cluster payoff/enabler hits (densities)
    primary = break_ties(cluster_stats)          # §7 tie-break
    return cluster_stats[primary]["copies"] / nonland_count(deck)

def ES_raw(deck, primary_cluster):
    # compute densities as per §5.2 for that cluster
    return compute_engine_support_density(deck, primary_cluster)

def CS_raw(deck):
    mvs = expand_nonland_mvs(deck)  # MV list with copies
    return stdev(mvs)

def MR_raw(deck):
    per_color_score = 0.0
    total_pips = sum_color_pips(deck)
    for c in colors_present(deck):
        sources = weighted_sources(deck, c)  # 1.0 / 0.9 / 0.8 / 0.5 rules
        target = target_for_color(deck, c)   # 14 / 12 / 8 rules
        share  = color_pip_share(deck, c, total_pips)
        per_c  = min(1.2, sources / max(1, target))
        per_color_score += per_c * share
    return per_color_score

def RD_raw(deck):
    depth = roles_with_depth(deck, roles={{"removal","countermagic","payoff","velocity"}}, min_depth=4)
    breadth = min(1.0, distinct_roles(deck) / 6.0)
    return 0.7 * (depth / 4.0) + 0.3 * breadth

def VEL_raw(deck):
    hits = count_velocity_hits(deck)  # MV<=2 draw; any MV scry/loot/connive/rummage/surveil
    return hits / nonland_count(deck)

def IF_raw(deck):
    cheap, total = count_interaction(deck)  # tag lexicon separates cheap vs total
    return cheap / max(1, total)

def MNSS(deck, baseline):
    # raw
    pc = PC_raw(deck)
    primary_cluster = identify_primary_cluster(deck)  # for ES
    es = ES_raw(deck, primary_cluster)
    cs = CS_raw(deck)
    mr = MR_raw(deck)
    rd = RD_raw(deck)
    vel = VEL_raw(deck)
    iff = IF_raw(deck)

    # normalize
    PC = map_z_to_unit(zscore(pc, baseline.mu["PC"], baseline.sigma["PC"]))
    ES = map_z_to_unit(zscore(es, baseline.mu["ES"], baseline.sigma["ES"]))
    # For CS, either compute z_CS directly and map once, or do the two-step below consistently:
    z_std = zscore(cs, baseline.mu["CS_raw"], baseline.sigma["CS_raw"])
    CS = map_z_to_unit(-z_std)
    MR = map_z_to_unit(zscore(mr, baseline.mu["MR"], baseline.sigma["MR"]))
    RD = map_z_to_unit(zscore(rd, baseline.mu["RD"], baseline.sigma["RD"]))
    VEL= map_z_to_unit(zscore(vel, baseline.mu["VEL"], baseline.sigma["VEL"]))
    IF = map_z_to_unit(zscore(iff, baseline.mu["IF"], baseline.sigma["IF"]))

    # aggregate
    return 100.0 * (0.25*PC + 0.20*ES + 0.10*CS + 0.15*MR + 0.10*RD + 0.10*VEL + 0.10*IF)
```

> Implementation note for CS: either  
> (a) compute `CS_raw = std_cmc`, then `z_std = zscore(CS_raw, μ_std, σ_std)` and set `z_CS = −z_std`, or  
> (b) compute `z_CS` directly and map to unit. Keep one approach consistently.

---

## 10) Sanity Checks & Calibration

- Centering: A deck with metric raw values equal to Baseline means should score near **50**.  
- Sensitivity: A +1σ improvement in a metric raises that metric’s contribution by +0.2 (20pp on the 0–1 scale) before weights.  
- Greedy manabases: MR should nudge greedy tri-color lists down unless they carry sufficient **weighted** sources.  
- Off-plan piles: PC and ES together should penalize unfocused builds.  
- Control vs Midrange: Control can score modestly in VEL (scry/loot) while ES rewards interaction densities without requiring raw draw volume to dominate.

---

## 11) Version History

- **v2.0**
  - Replaced ambiguous “norm(x)” with **map_z_to_unit(z)** and added σ floor.
  - CS uses **sign-inverted z of MV spread**; removed double-scaling language.
  - IF spec now includes zero-division guard.
  - MR adds **weighted sources** (1.0/0.9/0.8/0.5), clarified **targets (14/12/8)**, and pip-share logic.
  - ES redefined with **densities** per cluster; removed arbitrary denominators/caps.
  - VEL focuses on cheap smoothing (cantrips/scry/loot/connive/surveil).
  - RD emphasizes depth in plan-critical roles (+ modest breadth).
  - Documented **units** (densities over #nonland) and tied edge-case rulings to metrics.
  - Clarified **cluster tie-break** and priority order.

- **v1.x** (prior)
  - Initial MNSS specification and baseline calibration.
