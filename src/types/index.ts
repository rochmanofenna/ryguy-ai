export interface Command {
  name: string;
  description: string;
  handler: (args?: string[]) => void;
}

export interface TerminalLine {
  id: string;
  content: string;
  type: 'input' | 'output' | 'error' | 'system';
  timestamp: Date;
}

export interface Experience {
  id: string;
  title: string;
  company: string;
  duration: string;
  location: string;
  funding?: string;
  metrics: {
    throughput?: string;
    latency?: string;
    performance?: string;
    funding?: string;
  };
  bullets: {
    summary: string;
    technical: string;
    code_sample?: string;
  }[];
  demo_available: boolean;
  charts: string[];
}

export interface Project {
  id: string;
  name: string;
  status: 'RUNNING' | 'PUBLISHED' | 'DEMO READY' | 'COMPLETE' | 'SCALED' | 'DEPLOYED';
  badge?: string;
  description: string;
  metrics: string[];
  tech_stack: string[];
  demo_url?: string;
  code_url?: string;
}