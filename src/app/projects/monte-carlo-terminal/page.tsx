'use client';

import dynamic from 'next/dynamic';

const MonteCarloTerminal = dynamic(
  () => import('@/components/Projects/MonteCarlo/MonteCarloTerminal'),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl animate-pulse">Initializing Terminal...</div>
      </div>
    )
  }
);

export default function MonteCarloTerminalPage() {
  return <MonteCarloTerminal />;
}