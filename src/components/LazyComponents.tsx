'use client';

import { lazy, Suspense } from 'react';

// Loading component
const LoadingFallback = ({ name }: { name: string }) => (
  <div className="text-terminal-muted animate-pulse">
    Loading {name}...
  </div>
);

// Lazy load heavy visualizations
export const LazyTradingDashboard = ({ onInteraction }: any) => {
  const Component = lazy(() =>
    import('./PortfolioTerminal').then(module => ({
      default: module.TradingDashboard || (() => <div>Trading Dashboard</div>)
    }))
  );

  return (
    <Suspense fallback={<LoadingFallback name="Trading Dashboard" />}>
      <Component onInteraction={onInteraction} />
    </Suspense>
  );
};

export const LazyMonteCarloDemo = ({ onInteraction }: any) => {
  const Component = lazy(() =>
    import('./PortfolioTerminal').then(module => ({
      default: module.MonteCarloDemo || (() => <div>Monte Carlo Demo</div>)
    }))
  );

  return (
    <Suspense fallback={<LoadingFallback name="GPU Demo" />}>
      <Component onInteraction={onInteraction} />
    </Suspense>
  );
};

// Intersection Observer for viewport-based lazy loading
import { useEffect, useRef, useState } from 'react';

export const LazyLoadWrapper = ({
  children,
  threshold = 0.1,
  rootMargin = '50px'
}: {
  children: React.ReactNode;
  threshold?: number;
  rootMargin?: string;
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold, rootMargin }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [threshold, rootMargin]);

  return (
    <div ref={ref}>
      {isVisible ? children : <LoadingFallback name="content" />}
    </div>
  );
};

// Code syntax highlighter - lazy loaded
export const LazyCodeBlock = ({ code, language }: { code: string; language: string }) => {
  const [Prism, setPrism] = useState<any>(null);

  useEffect(() => {
    import('prismjs').then(module => {
      import(`prismjs/components/prism-${language}`).catch(() => {});
      setPrism(module.default);
    });
  }, [language]);

  useEffect(() => {
    if (Prism) {
      Prism.highlightAll();
    }
  }, [Prism, code]);

  if (!Prism) {
    return (
      <pre className="text-xs bg-black/30 p-2 rounded overflow-x-auto">
        <code>{code}</code>
      </pre>
    );
  }

  return (
    <pre className="text-xs bg-black/30 p-2 rounded overflow-x-auto">
      <code className={`language-${language}`}>{code}</code>
    </pre>
  );
};