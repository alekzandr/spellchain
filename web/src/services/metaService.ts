/**
 * MetaService - Handles fetching and caching Meta decklist data from the API.
 */

const API_BASE = 'http://localhost:8000';
const CACHE_KEY = 'spellchain_meta_cache';
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export type MetaDeckCard = {
    name: string;
    count: number;
    sideboard: boolean;
    roles: string[];
    card_data?: {
        mana_value: number;
        mana_cost: string;
        types: string[];
        colors: string[];
    };
}

export type MetaDeckResults = {
    raw: Record<string, number>;
    unit: Record<string, number>;
    mnss: number;
}

export type MetaDeck = {
    id: string;
    name: string;
    source: string;
    raw_list: string;
    filepath: string;
    results: MetaDeckResults | null;
    cards: MetaDeckCard[];
    error?: string;
}

export type MetaSummary = {
    deck_count: number;
    sources: Record<string, number>;
    avg_mnss: number;
    top_decks: Array<{
        id: string;
        name: string;
        source: string;
        mnss: number;
    }>;
}

export type MetaBaseline = {
    baseline: Record<string, { mu: number; sigma: number }>;
}

interface CachedData<T> {
    data: T;
    timestamp: number;
}

class MetaService {
    private cache: Map<string, CachedData<unknown>> = new Map();

    private async fetchWithCache<T>(
        endpoint: string,
        cacheKey: string,
        forceRefresh = false
    ): Promise<T> {
        // Check cache first
        if (!forceRefresh) {
            const cached = this.cache.get(cacheKey);
            if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
                return cached.data as T;
            }
        }

        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();
        this.cache.set(cacheKey, { data, timestamp: Date.now() });
        return data as T;
    }

    /**
     * Get all analyzed Meta decklists with MNSS scores.
     */
    async getMetaDecks(forceRefresh = false): Promise<{ decks: MetaDeck[]; count: number }> {
        return this.fetchWithCache('/meta/decks', 'meta_decks', forceRefresh);
    }

    /**
     * Get the baseline computed from Meta decklists.
     */
    async getMetaBaseline(forceRefresh = false): Promise<MetaBaseline> {
        return this.fetchWithCache('/meta/baseline', 'meta_baseline', forceRefresh);
    }

    /**
     * Get a summary of the current meta.
     */
    async getMetaSummary(forceRefresh = false): Promise<MetaSummary> {
        return this.fetchWithCache('/meta/summary', 'meta_summary', forceRefresh);
    }

    /**
     * Refresh the Meta analysis by reloading all decklists.
     */
    async refreshMeta(): Promise<{ status: string; deck_count: number; decks: MetaDeck[] }> {
        const response = await fetch(`${API_BASE}/meta/refresh`, { method: 'POST' });
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();

        // Clear cache on refresh
        this.cache.delete('meta_decks');
        this.cache.delete('meta_baseline');
        this.cache.delete('meta_summary');

        return data;
    }

    /**
     * Clear all cached data.
     */
    clearCache(): void {
        this.cache.clear();
    }

    /**
     * Get color for a deck based on its colors.
     */
    getDeckColor(deck: MetaDeck): string {
        if (!deck.cards.length) return 'hsl(0, 0%, 50%)';

        const colorCounts: Record<string, number> = {};
        for (const card of deck.cards) {
            if (card.card_data?.colors) {
                for (const color of card.card_data.colors) {
                    colorCounts[color] = (colorCounts[color] || 0) + card.count;
                }
            }
        }

        const colorMap: Record<string, string> = {
            W: 'hsl(45, 90%, 60%)',   // White - Gold
            U: 'hsl(210, 80%, 55%)',  // Blue
            B: 'hsl(280, 40%, 35%)',  // Black - Purple
            R: 'hsl(0, 75%, 55%)',    // Red
            G: 'hsl(120, 60%, 40%)',  // Green
        };

        const maxColor = Object.entries(colorCounts)
            .sort(([, a], [, b]) => b - a)[0];

        return maxColor ? colorMap[maxColor[0]] || 'hsl(0, 0%, 50%)' : 'hsl(0, 0%, 50%)';
    }

    /**
     * Get archetype from deck name.
     */
    getArchetype(deckName: string): string {
        const name = deckName.toLowerCase();
        if (name.includes('aggro')) return 'Aggro';
        if (name.includes('control')) return 'Control';
        if (name.includes('midrange')) return 'Midrange';
        if (name.includes('combo')) return 'Combo';
        if (name.includes('tempo')) return 'Tempo';
        if (name.includes('ramp')) return 'Ramp';
        return 'Unknown';
    }
}

export const metaService = new MetaService();
