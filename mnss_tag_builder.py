
#!/usr/bin/env python3
"""
mnss_tag_builder.py — Build a per-card lexicon (roles & clusters) from a card DB.

Purpose
-------
Generate a JSON "lexicon" that explicitly tags each card with roles (e.g. counter,
removal_hard, sweeper, ca, smoothing, enabler, payoff, threat) and optional clusters
(e.g. Azorius_Control). This lexicon can be fed into MNSS to avoid regex work each
time and to scale to larger formats (Standard, Pioneer, Modern) in batches.

Inputs
------
- --cards: Card database (CSV/TSV/pipe or JSON) with at least: name, mana_value (or cmc), types.
           Recommended: oracle_text, colors, mana_cost, tags.
- --config: Optional regex config (same schema as mnss_v2_config.json) to seed tagging.
            If omitted, a small built-in default is used.
- --clusters-from-decks: Optional directory of decklists (.txt) to infer cluster membership:
            For each deck, we assign all its nonland cards to a cluster by auto-detecting
            the best matching cluster via the config. This populates the "clusters" field in lexicon.

Outputs
-------
- --out: A JSON file with shape:
  {
    "cards": {
      "Card Name": {
        "roles": ["counter", "removal_hard", ...],
        "clusters": ["Azorius_Control", ...]   # optional
      },
      ...
    }
  }

Usage
-----
python mnss_tag_builder.py \
  --cards Standard_Cards.txt \
  --config mnss_v2_config.json \
  --clusters-from-decks decks/ \
  --out lexicon.json

You can run this per-format (Standard/Pioneer/Modern) and merge JSONs later.
"""

from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

# --------- Utils ---------
def load_text(p: Path) -> str:
    return p.read_text(encoding='utf-8', errors='ignore')

def sniff_delim(s: str) -> str:
    for d in ['|','\t',',',';']:
        if d in s: return d
    return ','

def normalize_name(s: str) -> str:
    import re as _re
    return _re.sub(r'\s+',' ', (s or '').strip()).lower()

# --------- Card DB ---------
def load_card_db(path: Path) -> Dict[str, Dict[str,Any]]:
    text = load_text(path)
    ext = path.suffix.lower()
    rows: List[Dict[str,Any]] = []
    if ext == '.json':
        data = json.loads(text)
        if isinstance(data,list): rows = data
        elif isinstance(data,dict) and 'cards' in data: rows = data['cards']
        else: raise ValueError('Unrecognized JSON card DB shape')
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
    db = {}
    for r in rows:
        nm = normalize_name(r.get('name',''))
        if nm: db[nm] = r
    return db

# --------- Config ---------
DEFAULT_CFG = {
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
  "clusters": {
    "Azorius_Control": {
      "includes": [r"\bcounter target\b", r"\bdestroy all\b|\bexile all\b", r"\bdraw\b|\bimpulse\b"],
      "excludes": []
    }
  }
}

def load_cfg(path: Optional[Path]) -> Dict[str,Any]:
    if not path: return DEFAULT_CFG
    ext = path.suffix.lower()
    try:
        if ext in ('.json',):
            data = json.loads(load_text(path))
        else:
            # lightweight YAML support if available
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(load_text(path))
            except Exception:
                data = DEFAULT_CFG
        cfg = json.loads(json.dumps(DEFAULT_CFG))
        for k,v in data.items():
            if isinstance(v,dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k]=v
        return cfg
    except Exception:
        return DEFAULT_CFG

# --------- Tagging ---------
def match_any(rx_list, text: str) -> bool:
    return any(re.search(rx, text, re.I) for rx in rx_list)

def roles_for_text(text: str, roles_rx: Dict[str, List[str]]) -> List[str]:
    labs = []
    for role, rx in roles_rx.items():
        if match_any(rx, text):
            labs.append(role)
    return labs

def read_decks_in_dir(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ('.txt','.deck','.dek')])

DECK_LINE_RE = re.compile(r'^\s*(?:(SB:)\s*)?(\d+)\s+(.+?)\s*$')
SIDEBOARD_HEADERS = re.compile(r'^\s*(sideboard|companions?)\s*$', re.I)

def parse_decklist(path: Path) -> List[Tuple[str,int]]:
    out = []
    in_side = False
    for raw in load_text(path).splitlines():
        s = raw.strip()
        if not s or s.startswith('#'): 
            continue
        if SIDEBOARD_HEADERS.match(s):
            in_side = True
            continue
        m = DECK_LINE_RE.match(s)
        if m:
            sb, count, name = m.groups()
            if sb or in_side: 
                continue  # ignore sideboard for cluster seeding
            out.append((name.strip(), int(count)))
    return out

def auto_cluster_for_deck(deck_cards: List[Tuple[str,int]], db: Dict[str,Dict[str,Any]], clusters_rx: Dict[str,Dict[str,List[str]]]) -> Optional[str]:
    best = None; best_val = -1
    for cname, cdef in clusters_rx.items():
        inc = cdef.get('includes', []); exc = cdef.get('excludes', [])
        score = 0
        for nm, ct in deck_cards:
            row = db.get(normalize_name(nm))
            if not row: continue
            body = f"{row.get('name','')}\n{row.get('oracle_text','')}\n{row.get('mana_cost','')}"
            if inc and not match_any(inc, body):
                continue
            if exc and match_any(exc, body):
                continue
            score += ct
        if score > best_val:
            best, best_val = cname, score
    return best

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build a per-card lexicon for MNSS")
    ap.add_argument('--cards', required=True, help='Card DB (csv/tsv/pipe/json)')
    ap.add_argument('--config', help='Regex config to seed tagging (json/yaml)')
    ap.add_argument('--clusters-from-decks', help='Directory of decklists (.txt) to infer cluster membership')
    ap.add_argument('--out', required=True, help='Output lexicon JSON')
    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.config)) if args.config else DEFAULT_CFG
    db = load_card_db(Path(args.cards))

    # Role tagging for every card
    lex_cards = {}
    for nm, row in db.items():
        text = f"{row.get('name','')}\n{row.get('mana_cost','')}\n{row.get('oracle_text','')}\n{row.get('types','')}"
        roles = roles_for_text(text, cfg.get('roles', {}))
        if roles:
            lex_cards[row.get('name','').strip()] = {"roles": sorted(set(roles))}

    # Optional: cluster tagging from decks
    if args.clusters_from_decks:
        d = Path(args.clusters_from_decks)
        deck_paths = read_decks_in_dir(d)
        for dp in deck_paths:
            deck_cards = parse_decklist(dp)
            cname = auto_cluster_for_deck(deck_cards, db, cfg.get('clusters', {}))
            if not cname: 
                continue
            # assign all nonland deck cards to that cluster (simple but effective)
            for nm, _ in deck_cards:
                row = db.get(normalize_name(nm))
                if not row: 
                    continue
                name_str = row.get('name','').strip()
                ent = lex_cards.setdefault(name_str, {"roles": []})
                ent.setdefault("clusters", [])
                if cname not in ent["clusters"]:
                    ent["clusters"].append(cname)

    out = {"cards": lex_cards}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"[ok] Wrote lexicon with {len(lex_cards)} tagged cards to {args.out}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
