import React from 'react';
import { motion } from 'framer-motion';
import type { MetaDeck } from '../services/metaService';

interface MetaPinwheelProps {
    decks: MetaDeck[];
    onDeckClick?: (deck: MetaDeck) => void;
    size?: number;
}

/**
 * Pinwheel visualization component for Meta decks.
 * Displays decks in a circular layout with MNSS scores.
 */
export const MetaPinwheel: React.FC<MetaPinwheelProps> = ({
    decks,
    onDeckClick,
    size = 320
}) => {
    if (!decks.length) {
        return (
            <div
                className="flex items-center justify-center bg-muted/20 rounded-full border-2 border-dashed border-muted"
                style={{ width: size, height: size }}
            >
                <p className="text-muted-foreground italic text-sm">No meta decks</p>
            </div>
        );
    }

    const validDecks = decks.filter(d => d.results);
    const totalDecks = validDecks.length;
    const anglePerDeck = 360 / totalDecks;
    const innerRadius = size * 0.2;
    const outerRadius = size * 0.42;
    const center = size / 2;

    // Color map for MTG colors
    const colorMap: Record<string, string> = {
        W: '#F9FAF4', // White
        U: '#0E68AB', // Blue
        B: '#150B00', // Black
        R: '#D3202A', // Red
        G: '#00733E', // Green
    };

    const getDeckColors = (deck: MetaDeck): string[] => {
        const colors = new Set<string>();
        for (const card of deck.cards) {
            if (card.card_data?.colors) {
                for (const c of card.card_data.colors) {
                    colors.add(c);
                }
            }
        }
        return Array.from(colors);
    };

    const getGradient = (deck: MetaDeck, index: number): string => {
        const colors = getDeckColors(deck);
        if (colors.length === 0) return 'hsl(0, 0%, 40%)';
        if (colors.length === 1) return colorMap[colors[0]] || 'hsl(0, 0%, 50%)';

        // Multi-color gradient
        const gradColors = colors.map(c => colorMap[c]).filter(Boolean);
        return `linear-gradient(${index * 45}deg, ${gradColors.join(', ')})`;
    };

    // Generate SVG path for each segment
    const createSegmentPath = (startAngle: number, endAngle: number): string => {
        const startRad = (startAngle - 90) * (Math.PI / 180);
        const endRad = (endAngle - 90) * (Math.PI / 180);

        const x1 = center + innerRadius * Math.cos(startRad);
        const y1 = center + innerRadius * Math.sin(startRad);
        const x2 = center + outerRadius * Math.cos(startRad);
        const y2 = center + outerRadius * Math.sin(startRad);
        const x3 = center + outerRadius * Math.cos(endRad);
        const y3 = center + outerRadius * Math.sin(endRad);
        const x4 = center + innerRadius * Math.cos(endRad);
        const y4 = center + innerRadius * Math.sin(endRad);

        const largeArc = endAngle - startAngle > 180 ? 1 : 0;

        return `M ${x1} ${y1} L ${x2} ${y2} A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${x3} ${y3} L ${x4} ${y4} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${x1} ${y1} Z`;
    };

    // Calculate label position for each segment
    const getLabelPosition = (index: number): { x: number; y: number; rotation: number } => {
        const midAngle = (index + 0.5) * anglePerDeck - 90;
        const labelRadius = (innerRadius + outerRadius) / 2;
        const rad = midAngle * (Math.PI / 180);

        return {
            x: center + labelRadius * Math.cos(rad),
            y: center + labelRadius * Math.sin(rad),
            rotation: midAngle + 90 > 90 && midAngle + 90 < 270 ? midAngle + 180 : midAngle
        };
    };

    return (
        <div className="relative" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="drop-shadow-xl">
                <defs>
                    {validDecks.map((deck, i) => {
                        const colors = getDeckColors(deck);
                        if (colors.length <= 1) return null;
                        return (
                            <linearGradient
                                key={`grad-${i}`}
                                id={`deckGrad-${i}`}
                                gradientTransform={`rotate(${i * 45})`}
                            >
                                {colors.map((c, ci) => (
                                    <stop
                                        key={ci}
                                        offset={`${(ci / (colors.length - 1)) * 100}%`}
                                        stopColor={colorMap[c] || '#666'}
                                    />
                                ))}
                            </linearGradient>
                        );
                    })}
                </defs>

                {/* Decorative outer ring */}
                <circle
                    cx={center}
                    cy={center}
                    r={outerRadius + 8}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="text-muted/30"
                />

                {/* Segments */}
                {validDecks.map((deck, i) => {
                    const startAngle = i * anglePerDeck;
                    const endAngle = (i + 1) * anglePerDeck - 2; // Small gap
                    const colors = getDeckColors(deck);
                    const fillColor = colors.length > 1
                        ? `url(#deckGrad-${i})`
                        : colors.length === 1
                            ? colorMap[colors[0]]
                            : '#666';

                    return (
                        <motion.path
                            key={deck.id}
                            d={createSegmentPath(startAngle, endAngle)}
                            fill={fillColor}
                            stroke="rgba(0,0,0,0.3)"
                            strokeWidth="1"
                            className="cursor-pointer"
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: i * 0.05, duration: 0.3 }}
                            whileHover={{
                                scale: 1.05,
                                filter: 'brightness(1.2)',
                                transition: { duration: 0.15 }
                            }}
                            onClick={() => onDeckClick?.(deck)}
                        />
                    );
                })}

                {/* Center circle */}
                <circle
                    cx={center}
                    cy={center}
                    r={innerRadius - 4}
                    className="fill-background stroke-muted"
                    strokeWidth="2"
                />
            </svg>

            {/* Center content */}
            <div
                className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none"
                style={{ padding: size * 0.35 }}
            >
                <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                    Meta
                </span>
                <span className="text-2xl font-black text-primary">
                    {totalDecks}
                </span>
                <span className="text-[10px] text-muted-foreground">
                    Decks
                </span>
            </div>

            {/* Deck labels - positioned outside */}
            <div className="absolute inset-0 pointer-events-none">
                {validDecks.map((deck, i) => {
                    const midAngle = (i + 0.5) * anglePerDeck - 90;
                    const labelRadius = outerRadius + 24;
                    const rad = midAngle * (Math.PI / 180);
                    const x = center + labelRadius * Math.cos(rad);
                    const y = center + labelRadius * Math.sin(rad);

                    // Determine text alignment based on position
                    const isRight = midAngle > -90 && midAngle < 90;

                    return (
                        <motion.div
                            key={deck.id}
                            className="absolute text-[9px] font-bold whitespace-nowrap"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.5 + i * 0.03 }}
                            style={{
                                left: x,
                                top: y,
                                transform: `translate(${isRight ? '0' : '-100%'}, -50%)`,
                            }}
                        >
                            <span className="text-muted-foreground">
                                {deck.name.length > 12 ? deck.name.slice(0, 12) + '…' : deck.name}
                            </span>
                            {deck.results && (
                                <span className="ml-1 text-primary font-black">
                                    {deck.results.mnss.toFixed(0)}
                                </span>
                            )}
                        </motion.div>
                    );
                })}
            </div>
        </div>
    );
};
