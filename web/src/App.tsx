import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { DeckGallery } from './pages/DeckGallery';
import { DeckImporter } from './pages/DeckImporter';
import { DeckDetail } from './pages/DeckDetail';
import { MetaSnapshot } from './pages/MetaSnapshot';

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-background text-foreground selection:bg-primary/30">
        <Sidebar />

        <main className="flex-1 p-8 overflow-y-auto">
          <div className="max-w-6xl mx-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/meta" element={<MetaSnapshot />} />
              <Route path="/decks" element={<DeckGallery />} />
              <Route path="/import" element={<DeckImporter />} />
              <Route path="/deck/:id" element={<DeckDetail />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
