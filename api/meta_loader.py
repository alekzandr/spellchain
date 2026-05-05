"""
meta_loader.py - Load and analyze Meta folder decklists with MNSS algorithm.

This module provides functionality to:
- Scan the Meta folder for decklist files
- Parse and analyze each decklist with MNSS
- Build a baseline from all Meta decks for normalization
- Return analyzed meta data with scores and card details
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from api.engine_wrapper import MNSSWrapper
import mnss_v2


class MetaLoader:
    """Handles loading and analyzing Meta folder decklists."""
    
    def __init__(self, meta_dir: str, card_db_path: str):
        """
        Initialize the MetaLoader.
        
        Args:
            meta_dir: Path to the Meta folder containing decklist files
            card_db_path: Path to the card database JSON file
        """
        self.meta_dir = Path(meta_dir)
        self.engine = MNSSWrapper(card_db_path)
        self._cached_decks: Optional[List[Dict[str, Any]]] = None
        self._cached_baseline: Optional[Dict[str, Dict[str, float]]] = None
    
    def get_decklist_files(self) -> List[Path]:
        """Get all .txt files from the Meta folder."""
        if not self.meta_dir.exists():
            return []
        return sorted(self.meta_dir.glob("*.txt"))
    
    def parse_deck_name(self, filepath: Path) -> Dict[str, str]:
        """
        Parse deck name and source from filename.
        
        Expected formats:
        - "MTG-Goldfish-Deck Name.txt" -> source: "MTG Goldfish", name: "Deck Name"
        - "Untapped-Deck Name.txt" -> source: "Untapped", name: "Deck Name"
        """
        filename = filepath.stem  # Remove .txt extension
        
        # Check for known sources
        if filename.startswith("MTG-Goldfish-"):
            source = "MTG Goldfish"
            name = filename[len("MTG-Goldfish-"):]
        elif filename.startswith("Untapped-"):
            source = "Untapped"
            name = filename[len("Untapped-"):]
        else:
            source = "Unknown"
            name = filename
        
        return {"name": name, "source": source}
    
    def load_decklist(self, filepath: Path) -> str:
        """Load decklist content from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_all_decklists(self) -> List[Dict[str, Any]]:
        """
        Load all decklists from the Meta folder.
        
        Returns:
            List of dicts containing deck info and raw content
        """
        decklists = []
        for filepath in self.get_decklist_files():
            info = self.parse_deck_name(filepath)
            content = self.load_decklist(filepath)
            decklists.append({
                "filepath": str(filepath),
                "name": info["name"],
                "source": info["source"],
                "raw_list": content
            })
        return decklists
    
    def build_baseline(self, force_refresh: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Build a baseline from all Meta decklists.
        
        Args:
            force_refresh: If True, rebuild baseline even if cached
            
        Returns:
            Baseline dict with mu and sigma for each metric
        """
        if self._cached_baseline and not force_refresh:
            return self._cached_baseline
        
        decklists = self.load_all_decklists()
        if not decklists:
            # Return dummy baseline if no decks
            return {k: {"mu": 0.0, "sigma": 1.0} for k in mnss_v2.RAW_METRICS.keys()}
        
        deck_strings = [d["raw_list"] for d in decklists]
        self._cached_baseline = self.engine.build_baseline_from_decks(deck_strings)
        return self._cached_baseline
    
    def analyze_all_decks(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Load and analyze all Meta decklists with MNSS.
        
        Args:
            force_refresh: If True, re-analyze even if cached
            
        Returns:
            List of analyzed deck data with MNSS scores
        """
        if self._cached_decks and not force_refresh:
            return self._cached_decks
        
        # First build the baseline from all decks
        baseline = self.build_baseline(force_refresh)
        
        # Load and analyze each deck
        decklists = self.load_all_decklists()
        analyzed_decks = []
        
        for deck_info in decklists:
            try:
                results = self.engine.score_deck(deck_info["raw_list"], baseline)
                analyzed_decks.append({
                    "id": deck_info["name"].lower().replace(" ", "-").replace("'", ""),
                    "name": deck_info["name"],
                    "source": deck_info["source"],
                    "raw_list": deck_info["raw_list"],
                    "filepath": deck_info["filepath"],
                    "results": {
                        "raw": results["raw"],
                        "unit": results["unit"],
                        "mnss": results["mnss"]
                    },
                    "cards": results["cards"]
                })
            except Exception as e:
                # Log error but continue with other decks
                print(f"Error analyzing deck {deck_info['name']}: {e}")
                analyzed_decks.append({
                    "id": deck_info["name"].lower().replace(" ", "-").replace("'", ""),
                    "name": deck_info["name"],
                    "source": deck_info["source"],
                    "raw_list": deck_info["raw_list"],
                    "filepath": deck_info["filepath"],
                    "error": str(e),
                    "results": None,
                    "cards": []
                })
        
        self._cached_decks = analyzed_decks
        return analyzed_decks
    
    def refresh(self) -> Dict[str, Any]:
        """
        Refresh the meta analysis by reloading all decklists.
        
        Returns:
            Dict with refresh status and deck count
        """
        self._cached_decks = None
        self._cached_baseline = None
        
        decks = self.analyze_all_decks(force_refresh=True)
        return {
            "status": "refreshed",
            "deck_count": len(decks),
            "decks": decks
        }
    
    def get_meta_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current meta.
        
        Returns:
            Dict with meta statistics
        """
        decks = self.analyze_all_decks()
        
        if not decks:
            return {
                "deck_count": 0,
                "sources": {},
                "avg_mnss": 0,
                "top_decks": []
            }
        
        # Count by source
        sources = {}
        valid_decks = [d for d in decks if d.get("results")]
        for deck in valid_decks:
            source = deck["source"]
            sources[source] = sources.get(source, 0) + 1
        
        # Calculate average MNSS
        mnss_scores = [d["results"]["mnss"] for d in valid_decks if d.get("results")]
        avg_mnss = sum(mnss_scores) / len(mnss_scores) if mnss_scores else 0
        
        # Get top decks by MNSS
        sorted_decks = sorted(valid_decks, key=lambda d: d["results"]["mnss"], reverse=True)
        top_decks = sorted_decks[:5]
        
        return {
            "deck_count": len(decks),
            "sources": sources,
            "avg_mnss": avg_mnss,
            "top_decks": [{
                "id": d["id"],
                "name": d["name"],
                "source": d["source"],
                "mnss": d["results"]["mnss"]
            } for d in top_decks]
        }
