import { useEffect, useState } from 'react';
import './App.css';
import { EvalPage } from './pages/EvalPage';
import { HandwritingPage } from './pages/HandwritingPage';
import { SpecPage } from './pages/SpecPage';
import { VoicePage } from './pages/VoicePage';

export default function App() {
  const [tab, setTab] = useState<'handwriting' | 'voice' | 'eval' | 'spec'>(() => {
    const saved = localStorage.getItem('mm.tab');
    if (saved === 'handwriting' || saved === 'voice' || saved === 'eval' || saved === 'spec') return saved;
    return 'handwriting';
  });

  useEffect(() => {
    localStorage.setItem('mm.tab', tab);
  }, [tab]);

  return (
    <div className="appShell">
      <header className="topBar">
        <div className="topBarTitle">
          <div className="topBarTitleMain">期末專題</div>
          <div className="topBarTitleSub">手寫數字辨識 / 語音數字辨識（純前端展示）</div>
        </div>
        <nav className="segmented" aria-label="功能選單">
          <button
            type="button"
            className={tab === 'handwriting' ? 'segmentedBtn active' : 'segmentedBtn'}
            onClick={() => setTab('handwriting')}
          >
            手寫辨識
          </button>
          <button
            type="button"
            className={tab === 'voice' ? 'segmentedBtn active' : 'segmentedBtn'}
            onClick={() => setTab('voice')}
          >
            語音辨識
          </button>
          <button
            type="button"
            className={tab === 'eval' ? 'segmentedBtn active' : 'segmentedBtn'}
            onClick={() => setTab('eval')}
          >
            準確度
          </button>
          <button
            type="button"
            className={tab === 'spec' ? 'segmentedBtn active' : 'segmentedBtn'}
            onClick={() => setTab('spec')}
          >
            專題說明
          </button>
        </nav>
      </header>

      <main className="content">
        {tab === 'handwriting' && <HandwritingPage />}
        {tab === 'voice' && <VoicePage />}
        {tab === 'eval' && <EvalPage />}
        {tab === 'spec' && <SpecPage />}
      </main>
    </div>
  );
}
