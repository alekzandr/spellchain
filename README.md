# Spellchain — MNSS v2 Deck Analyzer

Spellchain provides an implementation of MNSS (Meta‑Normalized Synergy Score) for evaluating Magic: The Gathering decks against a baseline metagame. It computes several deck quality metrics, normalizes them to a common 0–1 scale using z‑scores, and aggregates them into a single score on a 0–100 scale.

Core scripts:
- `mnss_v2.py`: Build a Baseline Meta from decks and score decks (single or batch).
- `mnss_tag_builder.py`: Build a per‑card lexicon (roles/clusters) to improve tagging quality and speed.
- `docs/MNSS_Design_Doc_v2.md`: Detailed metric and normalization spec.

## Features
- Weighted, normalized metrics: PC, ES, CS, MR, RD, VEL, IF.
- Baseline‑based or self‑normalized scoring (corpus / leave‑one‑out).
- Flexible inputs: CSV/TSV/pipe or JSON card databases; text decklists.
- Optional YAML/JSON config to customize clusters, roles, and land weights.
- Optional per‑card lexicon to override/augment regex‑based tagging.
- CSV summary and per‑deck JSON outputs.

## Requirements
- Python 3.9+
- Optional: `PyYAML` (for YAML configs). If not installed, JSON configs still work.

Recommended setup:
- Create a virtual environment and install `pyyaml` if you want YAML support.

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install pyyaml
```

## Inputs
- Card database (`--cards`):
  - JSON: list of card objects or `{ "cards": [...] }`.
  - Delimited text: CSV/TSV/pipe; header optional. Expected fields (case‑insensitive):
    - `name`, `mana_value` (or `cmc`/`manaValue`), `types`, `oracle_text` (recommended), `colors`, `mana_cost` (optional), `tags` (optional)
- Decklists:
  - Text files like `My Deck.txt` with lines in the form: `4 Card Name`
  - Sideboard lines are allowed and can be identified by `SB:` or a `Sideboard` header. Example:
    
    ```text
    4 Consider
    4 Make Disappear
    3 Memory Deluge
    2 The Wandering Emperor
    
    Sideboard
    2 Disdainful Stroke
    ```

## Quick Start

1) Build a Baseline Meta from a directory of decks

```bash
python mnss_v2.py build-baseline \
  --decks-dir decks/ \
  --cards Standard_Cards.txt \
  --out baseline_meta.json [--config cfg.json] [--lexicon lexicon.json]
```

2) Score deck(s) with a Baseline Meta

```bash
# Single deck
python mnss_v2.py score \
  --deck "Azorius Control.txt" \
  --cards Standard_Cards.txt \
  --baseline baseline_meta.json \
  --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]

# Directory of decks
python mnss_v2.py score \
  --decks-dir decks/ \
  --cards Standard_Cards.txt \
  --baseline baseline_meta.json \
  --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]
```

3) Self‑normalize (no external baseline)

```bash
# Normalize to the entire provided corpus
python mnss_v2.py score \
  --decks-dir decks/ \
  --cards Standard_Cards.txt \
  --self-norm corpus \
  --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]

# Leave‑one‑out (each deck scored against all others)
python mnss_v2.py score \
  --decks-dir decks/ \
  --cards Standard_Cards.txt \
  --self-norm loo \
  --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]
```

Outputs:
- Per‑deck: `out_v2/<deckname>.mnss.json` with raw/unit metrics and MNSS.
- Summary: `out_v2/summary.csv` with `MNSS`, `*_raw`, and `*_unit` columns.

## Configuration
Use JSON or YAML to customize:
- `clusters`: Named archetypes with `includes`/`excludes` regex lists.
- `roles`: Keyword/regex lists for tagging (e.g., `counter`, `removal_hard`, `sweeper`, etc.).
- `land_weights`: Tuples of `(regex, weight)` to weight colored sources for MR.
- `sigma_floor`: Guardrail for very small baseline standard deviations.

Get a starting template:

```bash
python mnss_v2.py --emit-example-config
```

Pass it via `--config cfg.json` on any command.

## Lexicon (Optional but Recommended)
A JSON lexicon lets you explicitly tag cards with `roles` and `clusters` to avoid relying only on regexes.

Build one from your card DB, optionally seeding clusters from a directory of decklists:

```bash
python mnss_tag_builder.py \
  --cards Standard_Cards.txt \
  --config mnss_v2_config.json \
  --clusters-from-decks decks/ \
  --out lexicon.json
```

Lexicon shape:

```json
{
  "cards": {
    "Card Name": {
      "roles": ["counter", "removal_hard", "sweeper", "ca", ...],
      "clusters": ["Azorius_Control", ...]
    }
  }
}
```

Use the lexicon with both baseline building and scoring via `--lexicon lexicon.json`.

## Metrics (High‑Level)
- PC: Primary cluster density (on‑plan maindeck copies).
- ES: Engine support density tailored to the primary cluster (e.g., cheap interaction ratio + velocity for control/spellchain).
- CS: Curve smoothness via non‑land mana value spread (lower spread → higher score).
- MR: Mana reliability using weighted color sources vs. colored pip demand.
- RD: Functional redundancy combining depth in critical roles and healthy breadth.
- VEL: Velocity from cheap card selection (draw/scry/loot/surveil/etc.).
- IF: Interaction fit (cheap interaction as a fraction of total interaction).

Full details and tie‑break rules are in `docs/MNSS_Design_Doc_v2.md`.

## Repo Layout
- `mnss_v2.py`: CLI for baseline building and deck scoring.
- `mnss_tag_builder.py`: CLI to generate a role/cluster lexicon from a card DB and decks.
- `docs/MNSS_Design_Doc_v2.md`: Specification for metrics and normalization.

## Tips & Notes
- Windows PowerShell: wrap paths with spaces in quotes, e.g., `--deck "Azorius Control.txt"`.
- Card DB flexibility: if your delimited file lacks a header, column order should be `name | mana_value | types | colors | tags | oracle_text`.
- Lands: the analyzer detects lands from the `types` field; sideboard entries are parsed but do not influence most metrics.
- Reproducibility: version your baseline (`baseline_meta.json`), config, and lexicon alongside deck results.

## Contributing
- Keep changes minimal and focused. Align with the design doc.
- If proposing new metrics or weight changes, include validation notes and sample baselines.

## License
TBD

## TODO (Spec v2 Alignment)
- ES: Implement cluster‑aware formulas (Spellchain/Control now; Landfall/Tokens/Graveyard next) using densities and ratios.
- Primary Cluster: Select by payoff+enabler density with tie‑breaks and priority order per doc.
- MR: Use pip‑share weighted aggregation and 1.2 per‑color cap; refine target thresholds based on double‑pip spells (by copies).
- RD: Use plan‑critical set {removal, countermagic, payoff, velocity}; compute `depth_count ≥ 4` and breadth `min(1, distinct/6)`.
- VEL: Count MV≤2 draw (role `ca`) OR any‑MV selection (role `smoothing`).
- IF: Replace diversity+exile bonus with `cheap / max(1, total)` interaction ratio.
- Roles/Config: Expand default roles to include `token_maker`, `landfall_payoff`, `fetch_like_enabler`, `gy_enabler` to support ES branches.
