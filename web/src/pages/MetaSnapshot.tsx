import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, TrendingUp, ExternalLink, Filter, ChevronDown } from 'lucide-react';
import { MetaPinwheel } from '../components/MetaPinwheel';
import { metaService, type MetaDeck, type MetaSummary } from '../services/metaService';
import { ManaCost } from '../components/ManaCost';

type SortOption = 'mnss' | 'name' | 'source';
type FilterSource = 'all' | 'MTG Goldfish' | 'Untapped';

export const MetaSnapshot: React.FC = () => {
    const [decks, setDecks] = useState<MetaDeck[]>([]);
    const [summary, setSummary] = useState<MetaSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedDeck, setSelectedDeck] = useState<MetaDeck | null>(null);
    const [sortBy, setSortBy] = useState<SortOption>('mnss');
    const [filterSource, setFilterSource] = useState<FilterSource>('all');

    const loadData = async (forceRefresh = false) => {
        try {
            setLoading(true);
            setError(null);
            const [decksResponse, summaryResponse] = await Promise.all([
                metaService.getMetaDecks(forceRefresh),
                metaService.getMetaSummary(forceRefresh)
            ]);
            setDecks(decksResponse.decks);
            setSummary(summaryResponse);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load meta data');
        } finally {
            setLoading(false);
        }
    };

    const handleRefresh = async () => {
        setRefreshing(true);
        try {
            await metaService.refreshMeta();
            await loadData(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to refresh');
        } finally {
            setRefreshing(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    // Filter and sort decks
    const filteredDecks = decks
        .filter(d => filterSource === 'all' || d.source === filterSource)
        .filter(d => d.results)
        .sort((a, b) => {
            switch (sortBy) {
                case 'mnss':
                    return (b.results?.mnss || 0) - (a.results?.mnss || 0);
                case 'name':
                    return a.name.localeCompare(b.name);
                case 'source':
                    return a.source.localeCompare(b.source);
                default:
                    return 0;
            }
        });

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6 text-center">
                <p className="text-red-400 font-bold mb-4">{error}</p>
                <button
                    onClick={() => loadData()}
                    className="px-4 py-2 bg-red-500 text-white rounded-xl font-bold hover:bg-red-600 transition-colors"
                >
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="space-y-12">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div>
                    <h2 className="text-4xl font-black tracking-tighter mb-2">Meta Snapshot</h2>
                    <p className="text-muted-foreground italic">
                        Current Standard meta analyzed with MNSS algorithm.
                    </p>
                </div>
                <button
                    onClick={handleRefresh}
                    disabled={refreshing}
                    className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-xl font-bold hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                    <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                    Refresh Meta
                </button>
            </div>

            {/* Summary Stats */}
            {summary && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-card border border-muted rounded-2xl p-6"
                    >
                        <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1">
                            Total Decks
                        </p>
                        <p className="text-3xl font-black">{summary.deck_count}</p>
                    </motion.div>
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="bg-card border border-muted rounded-2xl p-6"
                    >
                        <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1">
                            Avg. MNSS
                        </p>
                        <p className="text-3xl font-black text-primary">{summary.avg_mnss.toFixed(1)}%</p>
                    </motion.div>
                    {Object.entries(summary.sources).map(([source, count], i) => (
                        <motion.div
                            key={source}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 + i * 0.1 }}
                            className="bg-card border border-muted rounded-2xl p-6"
                        >
                            <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1">
                                {source}
                            </p>
                            <p className="text-3xl font-black">{count}</p>
                        </motion.div>
                    ))}
                </div>
            )}

            {/* Main Content: Pinwheel + List */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Pinwheel */}
                <div className="lg:col-span-1 flex flex-col items-center">
                    <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-primary" />
                        Meta Distribution
                    </h3>
                    <MetaPinwheel
                        decks={filteredDecks}
                        onDeckClick={setSelectedDeck}
                        size={300}
                    />
                </div>

                {/* Deck List */}
                <div className="lg:col-span-2">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-bold flex items-center gap-2">
                            <Filter className="w-5 h-5 text-primary" />
                            Meta Decklists
                        </h3>
                        <div className="flex items-center gap-3">
                            {/* Source Filter */}
                            <select
                                value={filterSource}
                                onChange={(e) => setFilterSource(e.target.value as FilterSource)}
                                className="px-3 py-2 bg-muted rounded-lg text-sm font-bold border border-muted appearance-none cursor-pointer"
                            >
                                <option value="all">All Sources</option>
                                <option value="MTG Goldfish">MTG Goldfish</option>
                                <option value="Untapped">Untapped</option>
                            </select>
                            {/* Sort */}
                            <select
                                value={sortBy}
                                onChange={(e) => setSortBy(e.target.value as SortOption)}
                                className="px-3 py-2 bg-muted rounded-lg text-sm font-bold border border-muted appearance-none cursor-pointer"
                            >
                                <option value="mnss">Sort by MNSS</option>
                                <option value="name">Sort by Name</option>
                                <option value="source">Sort by Source</option>
                            </select>
                        </div>
                    </div>

                    <div className="space-y-3">
                        {filteredDecks.map((deck, i) => (
                            <motion.div
                                key={deck.id}
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.03 }}
                                onClick={() => setSelectedDeck(deck)}
                                className={`bg-card/50 border rounded-2xl p-4 cursor-pointer transition-all hover:bg-muted/30 ${selectedDeck?.id === deck.id
                                    ? 'border-primary shadow-lg shadow-primary/10'
                                    : 'border-muted'
                                    }`}
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="w-12 h-12 bg-muted rounded-xl flex items-center justify-center font-black text-primary text-xl">
                                            {deck.results?.mnss.toFixed(0) || '—'}
                                        </div>
                                        <div>
                                            <h4 className="font-bold">{deck.name}</h4>
                                            <p className="text-xs text-muted-foreground">
                                                via {deck.source}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="flex gap-1">
                                            {deck.cards.slice(0, 3).map((c, ci) => (
                                                <div
                                                    key={ci}
                                                    className="text-[10px] bg-muted px-2 py-0.5 rounded text-muted-foreground"
                                                >
                                                    {c.name.split(' ')[0]}
                                                </div>
                                            ))}
                                        </div>
                                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Selected Deck Details */}
            {selectedDeck && selectedDeck.results && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-card border border-primary/30 rounded-3xl p-8"
                >
                    <div className="flex items-start justify-between mb-6">
                        <div>
                            <h3 className="text-2xl font-black">{selectedDeck.name}</h3>
                            <p className="text-sm text-muted-foreground">
                                Source: {selectedDeck.source}
                            </p>
                        </div>
                        <div className="text-right">
                            <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                                MNSS Score
                            </p>
                            <p className="text-4xl font-black text-primary">
                                {selectedDeck.results.mnss.toFixed(1)}%
                            </p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Metrics */}
                        <div>
                            <h4 className="font-bold mb-4 text-muted-foreground uppercase text-xs tracking-widest">
                                Raw Metrics
                            </h4>
                            <div className="grid grid-cols-2 gap-3">
                                {Object.entries(selectedDeck.results.raw).map(([key, value]) => (
                                    <div key={key} className="bg-muted/30 rounded-xl p-3">
                                        <p className="text-[10px] font-bold uppercase text-muted-foreground">
                                            {key}
                                        </p>
                                        <p className="text-lg font-black">
                                            {typeof value === 'number' ? value.toFixed(2) : value}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Card List */}
                        <div>
                            <h4 className="font-bold mb-4 text-muted-foreground uppercase text-xs tracking-widest">
                                Mainboard ({selectedDeck.cards.filter(c => !c.sideboard).length} cards)
                            </h4>
                            <div className="max-h-64 overflow-y-auto space-y-1 pr-2">
                                {selectedDeck.cards
                                    .filter(c => !c.sideboard)
                                    .map((card, i) => (
                                        <div
                                            key={i}
                                            className="flex items-center justify-between py-1 px-2 bg-muted/20 rounded-lg text-sm"
                                        >
                                            <div className="flex items-center gap-2">
                                                <span className="text-muted-foreground font-mono w-5">
                                                    {card.count}×
                                                </span>
                                                <span className="font-medium">{card.name}</span>
                                            </div>
                                            {card.card_data?.mana_cost && (
                                                <ManaCost cost={card.card_data.mana_cost} />
                                            )}
                                        </div>
                                    ))}
                            </div>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
};
