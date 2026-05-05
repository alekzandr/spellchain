import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Library, PlusCircle, Settings, Shield, Github, TrendingUp } from 'lucide-react';

export const Sidebar: React.FC = () => {
    return (
        <aside className="w-64 bg-card border-r border-muted flex flex-col h-screen sticky top-0 overflow-y-auto">
            <div className="p-8">
                <div className="flex items-center gap-3 mb-10 group cursor-pointer">
                    <div className="bg-primary p-2 rounded-xl group-hover:rotate-12 transition-transform shadow-lg shadow-primary/30">
                        <Shield className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-black tracking-tighter">Spellchain</h1>
                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">MNSS Engine v2</p>
                    </div>
                </div>

                <nav className="space-y-2">
                    <NavLink
                        to="/"
                        className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-bold group ${isActive ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'}`}
                    >
                        <LayoutDashboard className="w-5 h-5" />
                        Dashboard
                    </NavLink>
                    <NavLink
                        to="/meta"
                        className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-bold group ${isActive ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'}`}
                    >
                        <TrendingUp className="w-5 h-5" />
                        Meta Snapshot
                    </NavLink>
                    <NavLink
                        to="/decks"
                        className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-bold group ${isActive ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'}`}
                    >
                        <Library className="w-5 h-5" />
                        Deck Gallery
                    </NavLink>
                    <NavLink
                        to="/import"
                        className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl transition-all font-bold group ${isActive ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'}`}
                    >
                        <PlusCircle className="w-5 h-5" />
                        Import Deck
                    </NavLink>
                </nav>
            </div>

            <div className="mt-auto p-8 border-t border-muted">
                <div className="bg-muted/30 rounded-2xl p-4 space-y-4">
                    <div className="flex items-center gap-3 text-xs font-bold text-muted-foreground">
                        <Settings className="w-4 h-4" />
                        Settings
                    </div>
                    <a
                        href="#"
                        className="flex items-center gap-3 text-xs font-bold text-muted-foreground hover:text-primary transition-colors"
                    >
                        <Github className="w-4 h-4" />
                        Source Code
                    </a>
                </div>
            </div>
        </aside>
    );
};
