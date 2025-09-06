
#!/usr/bin/env python3
"""
mnss_v2.py — MNSS (Meta-Normalized Synergy Score) per Design Doc v2

Implements metrics: PC, ES, CS, MR, RD, VEL, IF
Normalization: z-scores vs Baseline Meta → map_z_to_unit
Weights: PC .25, ES .20, CS .10, MR .15, RD .10, VEL .10, IF .10

USAGE
=====
1) Build baseline from a directory of decks
   python mnss_v2.py build-baseline --decks-dir decks/ --cards Standard_Cards.txt --out baseline_meta.json [--config cfg.json]

2) Score (single or many) using a baseline
   python mnss_v2.py score --deck "Azorius Control.txt" --cards Standard_Cards.txt --baseline baseline_meta.json --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]
   python mnss_v2.py score --decks-dir decks/ --cards Standard_Cards.txt --baseline baseline_meta.json --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]

3) Self-normalize (no external baseline)
   python mnss_v2.py score --decks-dir decks/ --cards Standard_Cards.txt --self-norm corpus --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]
   python mnss_v2.py score --decks-dir decks/ --cards Standard_Cards.txt --self-norm loo    --out-dir out_v2 [--config cfg.json] [--lexicon lexicon.json]

CONFIG: same schema as mnss_v2_config.json (clusters, roles, land_weights, sigma_floor).
LEXICON: JSON from mnss_tag_builder.py {"cards": {"Card Name": {"roles":[...], "clusters":[...]}}}
"""

from __future__ import annotations
import argparse, csv, json, math, re, statistics, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ---------- Utils ----------
def load_text(p: Path) -> str:
    return p.read_text(encoding='utf-8', errors='ignore')

def sniff_delim(s: str) -> str:
    for d in ['|','\t',',',';']:
        if d in s: return d
    return ','

def normalize_name(s: str) -> str:
    return re.sub(r'\s+',' ', (s or '').strip()).lower()

def to_float(x: Any, default=0.0) -> float:
    try:
        if x is None: return default
        if isinstance(x,(int,float)): return float(x)
        s = str(x).strip()
        if not s: return default
        return float(s)
    except Exception:
        return default

# ---------- Deck & Card ----------
DECK_LINE_RE = re.compile(r'^\s*(?:(SB:)\s*)?(\d+)\s+(.+?)\s*$')
SIDEBOARD_HEADERS = re.compile(r'^\s*(sideboard|companions?)\s*$', re.I)

@dataclass
class DeckEntry:
    name: str
    count: int
    sideboard: bool=False

@dataclass(frozen=True)
class Card:
    name: str
    mana_value: float
    types: Tuple[str, ...]
    oracle_text: str
    colors: Tuple[str, ...]
    tags: Tuple[str, ...]
    mana_cost: str=''

def parse_decklist(path: Path) -> List[DeckEntry]:
    entries: List[DeckEntry] = []
    in_side=False
    for raw in load_text(path).splitlines():
        s = raw.strip()
        if not s or s.startswith('#'): continue
        if SIDEBOARD_HEADERS.match(s):
            in_side=True; continue
        m = DECK_LINE_RE.match(s)
        if m:
            sb, count, name = m.groups()
            entries.append(DeckEntry(name=name.strip(), count=int(float(count)), sideboard=bool(sb) or in_side))
    return entries

def load_card_db(path: Path) -> Dict[str, Card]:
    text = load_text(path); ext = path.suffix.lower()
    rows: List[Dict[str,Any]] = []
    if ext == '.json':
        data = json.loads(text)
        if isinstance(data,list): rows=data
        elif isinstance(data,dict) and 'cards' in data: rows=data['cards']
        else: raise ValueError('Unrecognized JSON card DB')
    else:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines: raise ValueError('Empty card DB')
        delim = sniff_delim(lines[0])
        has_header = any(h in lines[0].lower() for h in ['name','mana','types','cmc','oracle','colors','tags'])
        if has_header:
            rows = list(csv.DictReader(lines, delimiter=delim))
        else:
            # name | mana_value | types | colors | tags | oracle_text
            for parts in csv.reader(lines, delimiter=delim):
                parts = [p.strip() for p in parts]
                rows.append({
                    'name': parts[0] if len(parts)>0 else '',
                    'mana_value': parts[1] if len(parts)>1 else '',
                    'types': parts[2] if len(parts)>2 else '',
                    'colors': parts[3] if len(parts)>3 else '',
                    'tags': parts[4] if len(parts)>4 else '',
                    'oracle_text': parts[5] if len(parts)>5 else '',
                })
    def mk_card(row: Dict[str,Any]) -> Optional[Card]:
        name = (row.get('name') or '').strip()
        if not name: return None
        mv = row.get('mana_value', row.get('cmc', row.get('manaValue', 0)))
        types_field = row.get('types', row.get('type_line', row.get('type','')))
        types = tuple([t.strip() for t in re.split(r'[ /—-]+', str(types_field)) if t.strip()]) if types_field else tuple()
        oracle = str(row.get('oracle_text', row.get('text','')) or '')
        colors_field = row.get('colors', row.get('color_identity',''))
        if isinstance(colors_field,(list,tuple)):
            colors = tuple([str(c).strip().upper() for c in colors_field])
        else:
            colors = tuple([c.strip().upper() for c in re.split(r'[,/ ]+', str(colors_field)) if c.strip()])
        tags_field = row.get('tags','')
        if isinstance(tags_field,(list,tuple)):
            tags = tuple([str(t).strip().lower() for t in tags_field])
        else:
            tags = tuple([t.strip().lower() for t in re.split(r'[;,/ ]+', str(tags_field)) if t.strip()])
        mana_cost = str(row.get('mana_cost', row.get('cost','')) or '')
        return Card(name=name, mana_value=to_float(mv), types=types, oracle_text=oracle, colors=colors, tags=tags, mana_cost=mana_cost)
    db: Dict[str,Card] = {}
    for r in rows:
        c = mk_card(r)
        if c: db[normalize_name(c.name)] = c
    return db

# ---------- Tagging helpers ----------
def is_land(card: Card) -> bool:
    return 'land' in [t.lower() for t in card.types]

PIP_RE = re.compile(r'\{([WUBRG0-9X/]+)\}')
def estimate_pips(card: Card) -> int:
    s = card.mana_cost or card.oracle_text or ''
    total = 0
    for m in PIP_RE.finditer(s):
        tok = m.group(1)
        if any(c in tok for c in 'WUBRG'):
            total += 1
    return total

# ---------- Config ----------
DEFAULT_CFG = {
    "sigma_floor": 0.35,
    "primary_cluster": None,
    "clusters": {
        "Azorius_Control": {
            "includes": [r"\bcounter target\b", r"\bdestroy all\b|\bexile all\b", r"\bdraw\b|\bimpulse\b"],
            "excludes": []
        }
    },
    # TODO(spec v2): Consider expanding roles to include token_maker, landfall_payoff,
    # fetch_like_enabler, gy_enabler, interaction, etc., to support cluster-specific
    # Engine Support (ES) formulas.
    "roles": {
        "counter": [r"\bcounter target\b"],
        "removal_hard": [r"\bexile target\b", r"\bdestroy target\b(?!.*return)"],
        "removal_soft": [r"\breturn target\b.*\bowner'?s hand\b", r"\btap target\b(?!.*untap)"],
        "sweeper": [r"\bdestroy all\b|\bexile all\b|\beach creature\b"],
        "ca": [r"\bdraw\b|\bimpulse\b|\bspellbook\b|\blook at the top\b.*\binto your hand\b"],
        "smoothing": [r"\bscry\b|\bsurveil\b|\bloot\b|\bconnive\b|\bexplore\b"],
        "enabler": [r"\bspells you cast cost\b|\bwhenever you (?:draw|scry|surveil|connive)\b"],
        "payoff": [r"\bfor each\b|\bwhenever you draw\b|\bwhenever you cast\b"],
        "threat": [r"\bcreature\b|\bplaneswalker\b|\bcreate (?:a|two|\d+).+token\b"]
    },
    "land_weights": [
        ["\\benters the battlefield tapped\\b", 0.9],
        ["\\btapped unless\\b|\\bunless you control\\b|\\bif you control\\b|\\btwo or more\\b", 0.8],
        ["\\bscry 1\\b|\\bsurveil 1\\b", 0.95],
        ["\\bcreate a Treasure\\b|\\bTreasure\\b", 0.0],
        ["//|\\bmodal double-faced\\b|\\bDFC\\b", 0.5],
        [".*", 1.0]
    ]
}

def load_cfg(path: Optional[Path]) -> Dict[str,Any]:
    if not path: return DEFAULT_CFG
    try:
        if path.suffix.lower() in ('.yml','.yaml') and yaml is not None:
            data = yaml.safe_load(load_text(path))
        else:
            data = json.loads(load_text(path))
        cfg = json.loads(json.dumps(DEFAULT_CFG))
        for k,v in data.items():
            if isinstance(v,dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k]=v
        return cfg
    except Exception:
        return DEFAULT_CFG

# ---------- Optional Lexicon ----------
def load_lexicon(path: Optional[Path]) -> Dict[str, Any]:
    if not path: return {}
    try:
        data = json.loads(load_text(path))
        if isinstance(data, dict) and "cards" in data:
            return data
        return {}
    except Exception:
        return {}

def roles_from_lexicon(card: Card, lex: Dict[str, Any]) -> List[str]:
    if not lex: return []
    ent = lex.get("cards", {}).get(card.name, None)
    if ent and isinstance(ent, dict):
        return list(ent.get("roles", []))
    return []

def clusters_from_lexicon(card: Card, lex: Dict[str, Any]) -> List[str]:
    if not lex: return []
    ent = lex.get("cards", {}).get(card.name, None)
    if ent and isinstance(ent, dict):
        return list(ent.get("clusters", []))
    return []

def match_any(rx_list: List[str], text: str) -> bool:
    return any(re.search(rx, text, re.I) for rx in rx_list)

def card_has_role_name(card: Card, role_name: str, cfg_roles: Dict[str,List[str]], lex: Optional[Dict[str,Any]]) -> bool:
    # lexicon first
    if lex and role_name in set(roles_from_lexicon(card, lex)):
        return True
    # regex fallback
    rx = cfg_roles.get(role_name, [])
    body = f"{card.name}\n{card.mana_cost}\n{card.oracle_text}\n{' '.join(card.types)}"
    return match_any(rx, body)

def card_in_cluster(card: Card, cluster_name: str, cfg_clusters: Dict[str,Any], lex: Optional[Dict[str,Any]]) -> bool:
    if lex and cluster_name in set(clusters_from_lexicon(card, lex)):
        return True
    cdef = cfg_clusters.get(cluster_name, {"includes": [], "excludes": []})
    body = f"{card.name}\n{card.oracle_text}\n{card.mana_cost}"
    inc = cdef.get('includes', []); exc = cdef.get('excludes', [])
    if inc and not match_any(inc, body): return False
    if exc and match_any(exc, body): return False
    return True

# ---------- Metric raw calculators ----------
def mainboard_entries(entries: List[DeckEntry]) -> List[DeckEntry]:
    return [e for e in entries if not e.sideboard]

def counts_nonlands(entries: List[DeckEntry], db: Dict[str,Card]) -> Tuple[int,int]:
    nonlands=0; lands=0
    for e in entries:
        c = db.get(normalize_name(e.name))
        if not c: continue
        if is_land(c): lands += e.count
        else: nonlands += e.count
    return nonlands, lands

def mv_stats(entries: List[DeckEntry], db: Dict[str,Card]) -> Tuple[float,float]:
    arr = []
    for e in entries:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c): continue
        arr += [c.mana_value]*e.count
    if not arr: return 0.0, 0.0
    mu = statistics.mean(arr); sd = statistics.pstdev(arr)
    return mu, sd

def cluster_score(entries: List[DeckEntry], db: Dict[str,Card], cluster_name: str, cfg: Dict[str,Any], lex: Optional[Dict[str,Any]]) -> int:
    # TODO(spec v2): Return both payoff and enabler hits for density-based cluster
    # selection instead of a single summed score. Implement includes/excludes and
    # role-based counts, then compute densities in auto_primary_cluster with tie-breaks.
    score = 0
    for e in entries:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c): continue
        if card_in_cluster(c, cluster_name, cfg.get('clusters',{}), lex):
            score += e.count
    return score

def auto_primary_cluster(entries: List[DeckEntry], db: Dict[str,Card], cfg: Dict[str,Any], lex: Optional[Dict[str,Any]]) -> str:
    # TODO(spec v2): Choose by (payoff+enabler) density; tie-break by (payoff-enabler)
    # density; then by priority Spellchain > Landfall > Tokens > Control > Graveyard.
    best='Default'; best_val=-1
    for cname in cfg.get('clusters',{}).keys():
        val = cluster_score(entries, db, cname, cfg, lex)
        if val > best_val:
            best, best_val = cname, val
    return best

def metric_PC(entries, db, cfg, lex=None) -> float:
    main = mainboard_entries(entries)
    nonlands,_ = counts_nonlands(main, db)
    if nonlands == 0: return 0.0
    p_cluster = cfg.get('primary_cluster') or auto_primary_cluster(main, db, cfg, lex)
    matched = cluster_score(main, db, p_cluster, cfg, lex)
    return matched / nonlands

def metric_ES(entries, db, cfg, lex=None) -> float:
    main = mainboard_entries(entries)
    nonlands,_ = counts_nonlands(main, db)
    if nonlands == 0: return 0.0
    # TODO(spec v2): Compute ES per primary cluster using densities/ratios:
    # - Spellchain: 0.5*cheap_interaction_ratio + 0.5*velocity_density
    # - Control:   0.5*cheap_interaction_ratio + 0.5*min(1, interaction_density)
    # - Others (Landfall/Tokens/Graveyard): add branches per doc 5.2
    # Requires helpers: cheap/total interaction hits, velocity density (see VEL),
    # and primary cluster detection using tie-breaks.
    total = 0.0
    for e in main:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c): continue
        w = 1.0 if c.mana_value <= 3 else 0.7
        if card_has_role_name(c, 'enabler', cfg['roles'], lex): total += w*e.count
        if card_has_role_name(c, 'payoff',  cfg['roles'], lex): total += 1.1*w*e.count
    return total / nonlands

def metric_CS(entries, db, cfg, lex=None) -> float:
    _, sd = mv_stats(mainboard_entries(entries), db)
    # Curve smoothness uses the spread of non-land mana values; lower spread is better.
    # Return the standard deviation so sign inversion occurs only during normalization.
    return sd

def metric_MR(entries, db, cfg, lex=None) -> float:
    main = mainboard_entries(entries)
    # Demand
    pips = Counter(); early=set(); any_double=set()
    for e in main:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c): continue
        cp = estimate_pips(c)
        if cp >= 2:
            for col in c.colors: any_double.add(col)
        for col in c.colors: pips[col] += cp * e.count
        if c.mana_value <= 1:
            for col in c.colors: early.add(col)

    colors_in_deck = set([col for col,count in pips.items() if count>0])
    if not colors_in_deck: return 1.0

    # Sources
    def land_weight(card: Card) -> float:
        text = f"{card.name}\n{card.oracle_text}"
        for rx, w in cfg.get('land_weights', []):
            if re.search(rx, text, re.I): return float(w)
        return 1.0

    sources = Counter()
    for e in main:
        c = db.get(normalize_name(e.name))
        if not c or not is_land(c): continue
        w = land_weight(c)
        if not c.colors:
            name = c.name.lower()
            if 'plains' in name: cols=['W']
            elif 'island' in name: cols=['U']
            elif 'swamp' in name: cols=['B']
            elif 'mountain' in name: cols=['R']
            elif 'forest' in name: cols=['G']
            else: cols=[]
        else:
            cols = [cc for cc in c.colors]
        for col in cols:
            sources[col] += w * e.count

    # TODO(spec v2): Use double-pip spells counted by copies (not total pips):
    # if has early 1-drop OR >=8 double-pip spells -> 14
    # elif any double-pip spells -> 12
    # else -> 8
    def target_for(col: str) -> int:
        if (col in early) or (col in any_double and pips[col] >= 8): return 14
        elif (col in any_double) or (pips[col] >= 4): return 12
        else: return 8

    # TODO(spec v2): Aggregate with pip-share weighting and 1.2 cap per color:
    # per_c = min(1.2, sources[col] / max(1, target)); pip_share = pips[col]/sum_pips
    # MR_raw = sum(per_c * pip_share). Current implementation averages ratios equally.
    ratios = []
    for col in colors_in_deck:
        s = sources[col]; t = target_for(col)
        ratios.append(min(1.0, s/max(1.0, float(t))))
    return float(sum(ratios)/len(ratios)) if ratios else 1.0

def metric_RD(entries, db, cfg, lex=None) -> float:
    main = mainboard_entries(entries)
    nonlands,_ = counts_nonlands(main, db)
    if nonlands == 0: return 0.0
    counts = Counter()
    for e in main:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c): continue
        for r in ['counter','sweeper','removal_hard','removal_soft','ca','threat','smoothing']:
            if card_has_role_name(c, r, cfg['roles'], lex):
                counts[r] += e.count
    # TODO(spec v2): Redefine RD:
    # - Plan-critical roles = {removal (hard+soft+sweeper), countermagic, payoff, velocity}
    # - depth_count = #roles with depth >= 4 (by copies)
    # - breadth = min(1, distinct_roles / 6)
    # - RD_raw = 0.7*(depth_count/4) + 0.3*breadth
    critical = ['counter','sweeper','removal_hard']
    depth = sum(min(counts[r]/nonlands, 0.20) for r in critical)
    breadth = len([r for r,v in counts.items() if v>0]) / 7.0
    return 0.7*depth + 0.3*breadth

def metric_VEL(entries, db, cfg, lex=None) -> float:
    main = mainboard_entries(entries)
    nonlands,_ = counts_nonlands(main, db)
    if nonlands == 0: return 0.0
    total=0
    for e in main:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c): continue
        # Count VEL hits as either a cheap draw spell or any selection spell
        is_draw = card_has_role_name(c, 'ca', cfg['roles'], lex)
        is_smoothing = card_has_role_name(c, 'smoothing', cfg['roles'], lex)
        if (c.mana_value <= 2 and is_draw) or is_smoothing:
            total += e.count
    return total / nonlands

def metric_IF(entries, db, cfg, lex=None) -> float:
    main = mainboard_entries(entries)
    nonlands,_ = counts_nonlands(main, db)
    if nonlands == 0: return 0.0
    # Cheap interaction ratio per spec v2 (doc 5.7)
    cheap = 0
    total = 0
    for e in main:
        c = db.get(normalize_name(e.name))
        if not c or is_land(c):
            continue
        is_counter = card_has_role_name(c, 'counter', cfg['roles'], lex)
        is_hard = card_has_role_name(c, 'removal_hard', cfg['roles'], lex)
        is_soft = card_has_role_name(c, 'removal_soft', cfg['roles'], lex)
        is_sweeper = card_has_role_name(c, 'sweeper', cfg['roles'], lex)
        if is_counter or is_hard or is_soft or is_sweeper:
            total += e.count
            if c.mana_value <= 2 and (is_counter or is_hard or is_soft):
                cheap += e.count
    return cheap / max(1, total)

RAW_METRICS = {
    'PC': metric_PC, 'ES': metric_ES, 'CS': metric_CS, 'MR': metric_MR, 'RD': metric_RD, 'VEL': metric_VEL, 'IF': metric_IF
}
WEIGHTS = {'PC':0.25,'ES':0.20,'CS':0.10,'MR':0.15,'RD':0.10,'VEL':0.10,'IF':0.10}

# ---------- Normalization ----------
def map_z_to_unit(z: float) -> float:
    return max(0.0, min(1.0, 0.5 + 0.2*z))

def z_from(value: float, mu: float, sigma: float, sigma_floor: float) -> float:
    s = max(sigma, sigma_floor, 1e-6)
    return (value - mu) / s

def score_from_raw(raw: Dict[str,float], baseline: Dict[str,Dict[str,float]], sigma_floor: float) -> Tuple[Dict[str,float], float]:
    unit = {}
    for k, v in raw.items():
        bs = baseline.get(k, {'mu':0.0,'sigma':1.0})
        z = z_from(v, float(bs.get('mu',0.0)), float(bs.get('sigma',1.0)), sigma_floor)
        # Curve smoothness rewards lower spread, so invert its z-score here.
        unit[k] = map_z_to_unit(-z) if k=='CS' else map_z_to_unit(z)
    mnss = 100.0 * sum(WEIGHTS[k]*unit[k] for k in WEIGHTS)
    return unit, mnss

# ---------- Pipelines ----------
def compute_raw_metrics(deck_entries: List[DeckEntry], db: Dict[str,Card], cfg: Dict[str,Any], lex: Optional[Dict[str,Any]]=None) -> Dict[str,float]:
    return {k: fn(deck_entries, db, cfg, lex) for k, fn in RAW_METRICS.items()}

def read_decks_in_dir(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ('.txt','.deck','.dek')])

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def load_json(path: Path) -> Any:
    return json.loads(load_text(path))

def build_baseline_for_paths(deck_paths: List[Path], db: Dict[str,Card], cfg: Dict[str,Any], lex: Optional[Dict[str,Any]]=None) -> Dict[str,Dict[str,float]]:
    raws = {k: [] for k in RAW_METRICS.keys()}
    for dp in deck_paths:
        entries = parse_decklist(dp)
        raw = compute_raw_metrics(entries, db, cfg, lex)
        for k,v in raw.items():
            raws[k].append(v)
    baseline = {}
    for k, arr in raws.items():
        if not arr:
            baseline[k] = {'mu':0.0,'sigma':1.0}
        else:
            mu = statistics.mean(arr)
            sd = statistics.pstdev(arr) if len(arr)>1 else 1.0
            baseline[k] = {'mu':mu,'sigma':sd}
    return baseline

def score_deck_path(dp: Path, db: Dict[str,Card], cfg: Dict[str,Any], baseline: Dict[str,Dict[str,float]], out_dir: Optional[Path], lex: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    entries = parse_decklist(dp)
    raw = compute_raw_metrics(entries, db, cfg, lex)
    unit, mnss = score_from_raw(raw, baseline, cfg.get('sigma_floor', 0.35))
    result = {"deck": dp.name, "raw": raw, "unit": unit, "mnss": mnss}
    if out_dir: save_json(out_dir / f"{dp.stem}.mnss.json", result)
    return result

def corpus_baseline(deck_paths: List[Path], db: Dict[str,Card], cfg: Dict[str,Any], lex: Optional[Dict[str,Any]]=None) -> Dict[str,Dict[str,float]]:
    return build_baseline_for_paths(deck_paths, db, cfg, lex)

def loo_baseline(deck_paths: List[Path], db: Dict[str,Card], cfg: Dict[str,Any], exclude_idx: int, lex: Optional[Dict[str,Any]]=None) -> Dict[str,Dict[str,float]]:
    others = [p for i,p in enumerate(deck_paths) if i != exclude_idx]
    return build_baseline_for_paths(others, db, cfg, lex)

# ---------- CLI ----------
def emit_example_config():
    example = {
        "sigma_floor": 0.35,
        "primary_cluster": None,
        "clusters": DEFAULT_CFG["clusters"],
        "roles": DEFAULT_CFG["roles"],
        "land_weights": DEFAULT_CFG["land_weights"],
    }
    print(json.dumps(example, indent=2))

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MNSS v2 Analyzer")
    sub = ap.add_subparsers(dest="cmd", required=False)
    ap.add_argument('--config', help='Optional config (yaml/json) for clusters/roles/land weights')
    ap.add_argument('--emit-example-config', action='store_true', help='Print example config JSON and exit')

    sb = sub.add_parser('build-baseline', help='Build Baseline Meta from decks')
    sb.add_argument('--decks-dir', required=True)
    sb.add_argument('--cards', required=True)
    sb.add_argument('--out', required=True)
    sb.add_argument('--lexicon', help='Optional per-card lexicon JSON produced by mnss_tag_builder.py')

    ss = sub.add_parser('score', help='Score deck(s) with a baseline or self-normalization')
    ss.add_argument('--deck', help='Single deck path')
    ss.add_argument('--decks-dir', help='Directory of decks to score')
    ss.add_argument('--cards', required=True)
    ss.add_argument('--baseline', help='Baseline meta JSON (if omitted, use --self-norm)')
    ss.add_argument('--self-norm', choices=['corpus','loo'], help='Self-normalize over provided decks')
    ss.add_argument('--lexicon', help='Optional per-card lexicon JSON produced by mnss_tag_builder.py')
    ss.add_argument('--out-dir', required=True)

    args = ap.parse_args(argv)

    if args.emit_example_config:
        emit_example_config()
        return 0

    cfg = load_cfg(Path(args.config)) if args.config else DEFAULT_CFG

    if args.cmd == 'build-baseline':
        decks_dir = Path(args.decks_dir)
        deck_paths = read_decks_in_dir(decks_dir)
        if not deck_paths:
            print("[error] No decks found", file=sys.stderr); return 2
        db = load_card_db(Path(args.cards))
        lex = load_lexicon(Path(args.lexicon)) if args.lexicon else {}
        baseline = build_baseline_for_paths(deck_paths, db, cfg, lex)
        save_json(Path(args.out), baseline)
        print(f"[ok] Wrote baseline with metrics: {sorted(baseline.keys())} to {args.out}")
        return 0

    if args.cmd == 'score':
        db = load_card_db(Path(args.cards))
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        deck_paths: List[Path] = []
        if args.deck: deck_paths.append(Path(args.deck))
        if args.decks_dir: deck_paths += read_decks_in_dir(Path(args.decks_dir))
        if not deck_paths:
            print("[error] No deck(s) provided", file=sys.stderr); return 2
        lex = load_lexicon(Path(args.lexicon)) if args.lexicon else {}

        results = []
        if args.baseline:
            baseline = load_json(Path(args.baseline))
            for dp in deck_paths:
                results.append(score_deck_path(dp, db, cfg, baseline, out_dir, lex))
        else:
            if not args.self_norm:
                print("[error] Need --baseline or --self-norm", file=sys.stderr); return 2
            if args.self_norm == 'corpus':
                baseline = corpus_baseline(deck_paths, db, cfg, lex)
                for dp in deck_paths:
                    results.append(score_deck_path(dp, db, cfg, baseline, out_dir, lex))
            else:
                for i, dp in enumerate(deck_paths):
                    bl = loo_baseline(deck_paths, db, cfg, exclude_idx=i, lex=lex)
                    results.append(score_deck_path(dp, db, cfg, bl, out_dir, lex))

        # CSV summary
        with (out_dir / "summary.csv").open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ["deck","MNSS"] + [f"{k}_raw" for k in RAW_METRICS.keys()] + [f"{k}_unit" for k in RAW_METRICS.keys()]
            w.writerow(header)
            for r in results:
                w.writerow([r["deck"], f"{r['mnss']:.2f}"] + [f"{r['raw'][k]:.6f}" for k in RAW_METRICS] + [f"{r['unit'][k]:.4f}" for k in RAW_METRICS])
        print(f"[ok] Scored {len(results)} deck(s). CSV at: {out_dir/'summary.csv'}")
        return 0

    ap.print_help()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
