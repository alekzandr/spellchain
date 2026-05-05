import React, { useEffect, useState } from 'react';
import { storage } from '../utils/storage';
import { TrendingUp, Activity, Database, Clock, RefreshCw } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { MetaPinwheel } from '../components/MetaPinwheel';
import { metaService, type MetaDeck, type MetaSummary } from '../services/metaService';

export const Dashboard: React.FC = () => {
    const decks = storage.getDecks();
    const recentDecks = [...decks].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()).slice(0, 3);
    const avgMnss = decks.length > 0 ? decks.reduce((a, b) => a + b.results.mnss, 0) / decks.length : 0;

    // Meta state
    const [metaDecks, setMetaDecks] = useState<MetaDeck[]>([]);
    const [metaSummary, setMetaSummary] = useState<MetaSummary | null>(null);
    const [metaLoading, setMetaLoading] = useState(true);
    const [metaError, setMetaError] = useState<string | null>(null);

    useEffect(() => {
        const loadMeta = async () => {
            try {
                setMetaLoading(true);
                const [decksRes, summaryRes] = await Promise.all([
                    metaService.getMetaDecks(),
                    metaService.getMetaSummary()
                ]);
                setMetaDecks(decksRes.decks);
                setMetaSummary(summaryRes);
            } catch (err) {
                setMetaError(err instanceof Error ? err.message : 'Failed to load meta');
            } finally {
                setMetaLoading(false);
            }
        };
        loadMeta();
    }, []);

    const stats = [
        { label: "Decks Analyzed", value: decks.length, icon: Database, color: "text-primary" },
        { label: "Avg. Synergy", value: `${avgMnss.toFixed(1)}%`, icon: TrendingUp, color: "text-accent" },
        { label: "Meta Decks", value: metaSummary?.deck_count || 0, icon: Activity, color: "text-green-500" },
    ];

    return (
        <div className="space-y-12">
            <div>
                <h2 className="text-4xl font-black tracking-tighter mb-2">Welcome, Decksmith</h2>
                <p className="text-muted-foreground italic">Your central hub for meta-normalized synergy analysis.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {stats.map((stat, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        className="bg-card border border-muted rounded-3xl p-8 flex flex-col gap-4 shadow-xl"
                    >
                        <div className={`p-3 bg-muted w-fit rounded-2xl ${stat.color}`}>
                            <stat.icon className="w-6 h-6" />
                        </div>
                        <div>
                            <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">{stat.label}</p>
                            <p className="text-3xl font-black">{stat.value}</p>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Meta Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="bg-card border border-muted rounded-3xl p-8"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-primary" />
                            Current Meta
                        </h3>
                        <Link to="/meta" className="text-sm font-bold text-primary hover:underline">
                            View All &rarr;
                        </Link>
                    </div>

                    {metaLoading ? (
                        <div className="flex items-center justify-center h-64">
                            <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
                        </div>
                    ) : metaError ? (
                        <div className="flex items-center justify-center h-64 text-muted-foreground">
                            <p className="text-sm italic">{metaError}</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center">
                            <MetaPinwheel decks={metaDecks.slice(0, 8)} size={240} />
                            {metaSummary && (
                                <div className="mt-6 text-center">
                                    <p className="text-sm text-muted-foreground">
                                        <span className="font-bold text-foreground">{metaSummary.deck_count}</span> decks from{' '}
                                        {Object.keys(metaSummary.sources).join(' & ')}
                                    </p>
                                    <p className="text-xs text-muted-foreground mt-1">
                                        Avg. MNSS: <span className="font-bold text-primary">{metaSummary.avg_mnss.toFixed(1)}%</span>
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </motion.div>

                {/* Top Meta Decks */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="bg-card border border-muted rounded-3xl p-8"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Activity className="w-5 h-5 text-green-500" />
                            Top Meta Decks
                        </h3>
                    </div>

                    {metaLoading ? (
                        <div className="flex items-center justify-center h-48">
                            <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
                        </div>
                    ) : metaSummary?.top_decks.length ? (
                        <div className="space-y-3">
                            {metaSummary.top_decks.slice(0, 5).map((deck, i) => (
                                <Link key={deck.id} to="/meta">
                                    <motion.div
                                        initial={{ opacity: 0, x: 20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: 0.5 + i * 0.05 }}
                                        className="flex items-center justify-between p-3 bg-muted/30 rounded-xl hover:bg-muted/50 transition-colors group"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 bg-muted rounded-lg flex items-center justify-center font-black text-primary text-sm">
                                                {deck.mnss.toFixed(0)}
                                            </div>
                                            <div>
                                                <p className="font-bold text-sm group-hover:text-primary transition-colors">{deck.name}</p>
                                                <p className="text-[10px] text-muted-foreground">{deck.source}</p>
                                            </div>
                                        </div>
                                        <div className="text-xs text-muted-foreground">#{i + 1}</div>
                                    </motion.div>
                                </Link>
                            ))}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center h-48 text-muted-foreground">
                            <p className="text-sm italic">No meta decks available</p>
                        </div>
                    )}
                </motion.div>
            </div>

            {/* Recent Activity */}
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <Clock className="w-5 h-5 text-primary" />
                        Recent Activity
                    </h3>
                    <Link to="/decks" className="text-sm font-bold text-primary hover:underline">View All &rarr;</Link>
                </div>

                <div className="grid grid-cols-1 gap-4">
                    {recentDecks.length > 0 ? (
                        recentDecks.map((deck) => (
                            <Link key={deck.id} to={`/deck/${deck.id}`}>
                                <div className="bg-card/50 border border-muted rounded-2xl p-6 hover:bg-muted/30 transition-all flex items-center justify-between group">
                                    <div className="flex items-center gap-6">
                                        <div className="w-12 h-12 bg-muted rounded-xl flex items-center justify-center font-black text-primary text-xl">
                                            {deck.results.mnss.toFixed(0)}
                                        </div>
                                        <div>
                                            <h4 className="font-bold group-hover:text-primary transition-colors">{deck.name}</h4>
                                            <p className="text-xs text-muted-foreground">{new Date(deck.created_at).toLocaleDateString()}</p>
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        {deck.cards.slice(0, 5).map((c, i) => (
                                            <div key={i} className="text-[10px] bg-muted px-2 py-0.5 rounded text-muted-foreground border border-muted-foreground/10">
                                                {c.name.split(' ')[0]}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </Link>
                        ))
                    ) : (
                        <div className="bg-muted/10 border-2 border-dashed border-muted rounded-2xl p-12 text-center">
                            <p className="text-muted-foreground italic">No recent activity. Import a deck to get started!</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
