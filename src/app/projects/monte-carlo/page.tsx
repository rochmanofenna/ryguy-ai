import dynamic from 'next/dynamic';

const MonteCarloDemo = dynamic(
  () => import('@/components/Projects/MonteCarlo/MonteCarloDemo'),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl animate-pulse">Loading Monte Carlo Demo...</div>
      </div>
    )
  }
);

export default function MonteCarloPage() {
  return <MonteCarloDemo />;
}