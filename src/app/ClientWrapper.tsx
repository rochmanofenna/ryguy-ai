'use client';

import dynamic from 'next/dynamic';

const PortfolioTerminal = dynamic(
  () => import('@/components/PortfolioTerminal'),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-[#0A0E1B] text-[#E8E9ED] p-4 sm:p-6 md:p-8">
        <div className="max-w-5xl mx-auto">
          <div className="font-mono text-sm animate-pulse">
            Initializing terminal...
          </div>
        </div>
      </div>
    )
  }
);

export default function ClientWrapper() {
  return <PortfolioTerminal />;
}