'use client';

import { useState, useEffect } from 'react';
import Terminal from './Terminal';
import TerminalEnhanced from './TerminalEnhanced';

/**
 * Smart Terminal Component
 * Automatically detects if BEF service is available and uses the enhanced version,
 * otherwise falls back to the original terminal with mock data
 */
export default function SmartTerminal() {
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(null);
  const [checkingService, setCheckingService] = useState(true);

  useEffect(() => {
    // Check if BEF service is running
    const checkService = async () => {
      try {
        const response = await fetch('http://localhost:8000/', {
          method: 'GET',
          mode: 'cors',
          signal: AbortSignal.timeout(2000) // 2 second timeout
        });

        if (response.ok) {
          const data = await response.json();
          if (data.service === 'BEF Pipeline Service') {
            setServiceAvailable(true);
            console.log('✅ BEF Pipeline Service detected - using enhanced terminal');
            return;
          }
        }
      } catch (error) {
        console.log('⚠️ BEF service not available - using standard terminal');
      }

      setServiceAvailable(false);
      setCheckingService(false);
    };

    checkService();

    // Re-check every 10 seconds if service is not available
    const interval = setInterval(() => {
      if (!serviceAvailable) {
        checkService();
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [serviceAvailable]);

  // Show loading state while checking
  if (serviceAvailable === null && checkingService) {
    return (
      <div className="min-h-screen bg-terminal-bg text-terminal-text p-4 sm:p-6 md:p-8">
        <div className="max-w-5xl mx-auto font-mono text-xs sm:text-sm">
          <div className="animate-pulse">
            <div className="text-terminal-accent">Initializing terminal...</div>
            <div className="text-terminal-muted mt-2">Checking BEF Pipeline Service availability...</div>
          </div>
        </div>
      </div>
    );
  }

  // Use enhanced terminal if service is available
  if (serviceAvailable) {
    return (
      <>
        <TerminalEnhanced />
        <ServiceIndicator status="connected" />
      </>
    );
  }

  // Fall back to original terminal
  return (
    <>
      <Terminal />
      <ServiceIndicator status="offline" />
    </>
  );
}

// Service status indicator
function ServiceIndicator({ status }: { status: 'connected' | 'offline' }) {
  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-terminal-bg border ${
        status === 'connected' ? 'border-green-500/30' : 'border-yellow-500/30'
      }`}>
        <div className={`w-2 h-2 rounded-full ${
          status === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'
        }`} />
        <span className="text-terminal-muted text-xs font-mono">
          {status === 'connected' ? 'BEF Pipeline Active' : 'Simulation Mode'}
        </span>
      </div>
      {status === 'offline' && (
        <div className="mt-2 text-right">
          <span className="text-terminal-muted text-xs font-mono opacity-60">
            Run: ./start_bef_service.sh
          </span>
        </div>
      )}
    </div>
  );
}