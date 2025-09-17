/**
 * BEF Pipeline Service Client
 * Connects React Terminal to the real BICEP, ENN, and FusionAlpha implementations
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export interface PathGenerationRequest {
  n_paths: number;
  n_steps: number;
  T?: number;
  feedback_value?: number;
  decay_rate?: number;
  method?: 'euler_maruyama' | 'milstein' | 'heun';
}

export interface PathGenerationResult {
  paths_shape: number[];
  statistics: {
    mean_final: number;
    std_final: number;
    max_value: number;
    min_value: number;
    generation_time_ms: number;
    paths_per_second: number;
  };
  sample_path: number[];
  timestamp: string;
}

export interface ENNPredictionRequest {
  sequence: number[];
  use_entanglement?: boolean;
  lambda_param?: number;
}

export interface ENNPredictionResult {
  sequence_length: number;
  final_prediction: number;
  confidence: number;
  entanglement_eigenvalues: number[];
  hidden_state_norm: number;
  psi_evolution: number[];
  timestamp: string;
}

export interface FusionAlphaRequest {
  enn_predictions: Array<{
    committor: number;
    confidence: number;
  }>;
  use_severity_scaling?: boolean;
  confidence_threshold?: number;
}

export interface FusionAlphaResult {
  num_nodes: number;
  decisions: Array<{
    node: string;
    committor: number;
    decision: number;
    confidence: number;
  }>;
  average_committor: number;
  decision_distribution: number;
  severity_scaling_applied: boolean;
  timestamp: string;
}

export interface PortfolioMetrics {
  current_pnl: number;
  daily_pnl: number;
  sharpe_ratio: number;
  calmar_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  var_95: number;
  current_exposure: number;
  beta: number;
}

export interface GPUBenchmark {
  benchmarks: Array<{
    paths: number;
    time_ms: number;
    paths_per_second: number;
    speedup: number;
  }>;
  device: string;
  precision: string;
  timestamp: string;
}

class BEFService {
  private wsConnection: WebSocket | null = null;
  private wsListeners: Set<(data: any) => void> = new Set();

  /**
   * Generate stochastic paths using BICEP engine
   */
  async generatePaths(request: PathGenerationRequest): Promise<PathGenerationResult> {
    const response = await fetch(`${API_BASE_URL}/api/bicep/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`BICEP generation failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get ENN prediction for a sequence
   */
  async predictWithENN(request: ENNPredictionRequest): Promise<ENNPredictionResult> {
    const response = await fetch(`${API_BASE_URL}/api/enn/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`ENN prediction failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Run FusionAlpha planning with severity scaling
   */
  async planWithFusionAlpha(request: FusionAlphaRequest): Promise<FusionAlphaResult> {
    const response = await fetch(`${API_BASE_URL}/api/fusion/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`FusionAlpha planning failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get real-time portfolio metrics
   */
  async getPortfolioMetrics(): Promise<PortfolioMetrics> {
    const response = await fetch(`${API_BASE_URL}/api/portfolio/metrics`);

    if (!response.ok) {
      throw new Error(`Failed to fetch portfolio metrics: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Run GPU benchmark
   */
  async runGPUBenchmark(): Promise<GPUBenchmark> {
    const response = await fetch(`${API_BASE_URL}/api/gpu/benchmark`);

    if (!response.ok) {
      throw new Error(`GPU benchmark failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Connect to WebSocket for real-time streaming
   */
  connectWebSocket(onMessage: (data: any) => void): void {
    if (this.wsConnection?.readyState === WebSocket.OPEN) {
      this.wsListeners.add(onMessage);
      return;
    }

    this.wsConnection = new WebSocket(`${WS_URL}/ws/stream`);
    this.wsListeners.add(onMessage);

    this.wsConnection.onopen = () => {
      console.log('Connected to BEF pipeline stream');
    };

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.wsListeners.forEach(listener => listener(data));
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.wsConnection.onclose = () => {
      console.log('Disconnected from BEF pipeline stream');
      this.wsConnection = null;
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (this.wsListeners.size > 0) {
          this.wsListeners.forEach(listener => this.connectWebSocket(listener));
        }
      }, 5000);
    };
  }

  /**
   * Disconnect from WebSocket
   */
  disconnectWebSocket(onMessage?: (data: any) => void): void {
    if (onMessage) {
      this.wsListeners.delete(onMessage);
    }

    if (this.wsListeners.size === 0 && this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Run complete BEF pipeline (BICEP -> ENN -> FusionAlpha)
   */
  async runCompletePipeline(n_sequences: number = 100): Promise<{
    bicep: PathGenerationResult;
    enn: ENNPredictionResult;
    fusion: FusionAlphaResult;
  }> {
    // Step 1: Generate paths with BICEP
    const bicepResult = await this.generatePaths({
      n_paths: n_sequences,
      n_steps: 100,
      T: 1.0,
      feedback_value: 0.7,
      decay_rate: 0.1,
      method: 'euler_maruyama'
    });

    // Step 2: Use sample path for ENN prediction
    const samplePath = bicepResult.sample_path.slice(0, 20);
    const ennResult = await this.predictWithENN({
      sequence: samplePath,
      use_entanglement: true,
      lambda_param: 0.1
    });

    // Step 3: Use ENN predictions for FusionAlpha planning
    const fusionResult = await this.planWithFusionAlpha({
      enn_predictions: [
        {
          committor: ennResult.final_prediction,
          confidence: ennResult.confidence
        }
      ],
      use_severity_scaling: true,
      confidence_threshold: 0.5
    });

    return {
      bicep: bicepResult,
      enn: ennResult,
      fusion: fusionResult
    };
  }
}

// Export singleton instance
export const befService = new BEFService();

// Export for use in terminal commands
export default befService;