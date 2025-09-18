// WebGPU Monte Carlo Option Pricing Engine
// High-performance GPU implementation for Black-Scholes Monte Carlo simulation

/// <reference types="@webgpu/types" />

export interface OptionParams {
  S: number;     // Current stock price
  K: number;     // Strike price
  r: number;     // Risk-free rate
  T: number;     // Time to maturity (years)
  sigma: number; // Volatility
  paths: number; // Number of Monte Carlo paths
  steps: number; // Time steps per path
}

export interface PricingResult {
  price: number;
  standardError: number;
  executionTime: number;
  pathsPerSecond: number;
  confidence95: [number, number];
  device: 'webgpu' | 'cpu' | 'gpu.js' | 'webgl';
}

export class MonteCarloWebGPU {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  async initialize(): Promise<boolean> {
    if (!navigator.gpu) {
      console.warn('WebGPU not supported, falling back to CPU');
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter();
      if (!this.adapter) return false;

      this.device = await this.adapter.requestDevice();

      // Create compute shader
      const shaderModule = this.device.createShaderModule({
        label: 'Monte Carlo Compute Shader',
        code: this.getShaderCode()
      });

      // Create bind group layout
      this.bindGroupLayout = this.device.createBindGroupLayout({
        label: 'Monte Carlo Bind Group Layout',
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
          }
        ]
      });

      // Create compute pipeline
      this.computePipeline = this.device.createComputePipeline({
        label: 'Monte Carlo Pipeline',
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [this.bindGroupLayout]
        }),
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });

      return true;
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
      return false;
    }
  }

  private getShaderCode(): string {
    return `
      struct Params {
        S: f32,         // Current stock price
        K: f32,         // Strike price
        r: f32,         // Risk-free rate
        T: f32,         // Time to maturity
        sigma: f32,     // Volatility
        paths: u32,     // Number of paths
        steps: u32,     // Steps per path
        seed: u32,      // Random seed
      }

      @group(0) @binding(0) var<storage, read> params: Params;
      @group(0) @binding(1) var<storage, read_write> results: array<f32>;

      // Linear Congruential Generator for pseudo-random numbers
      fn lcg(seed: ptr<function, u32>) -> f32 {
        let a = 1664525u;
        let c = 1013904223u;
        *seed = a * (*seed) + c;
        return f32(*seed) / 4294967296.0;
      }

      // Box-Muller transform for normal distribution
      fn boxMuller(seed: ptr<function, u32>) -> vec2<f32> {
        let u1 = lcg(seed);
        let u2 = lcg(seed);
        let r = sqrt(-2.0 * log(max(u1, 1e-10)));
        let theta = 2.0 * 3.14159265359 * u2;
        return vec2<f32>(r * cos(theta), r * sin(theta));
      }

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= params.paths) {
          return;
        }

        var seed = params.seed + idx;
        let dt = params.T / f32(params.steps);
        let drift = (params.r - 0.5 * params.sigma * params.sigma) * dt;
        let diffusion = params.sigma * sqrt(dt);

        var price = params.S;

        // Simulate path
        for (var i = 0u; i < params.steps; i = i + 1u) {
          let z = boxMuller(&seed).x;
          price = price * exp(drift + diffusion * z);
        }

        // Calculate payoff (European call option)
        let payoff = max(price - params.K, 0.0);
        results[idx] = payoff * exp(-params.r * params.T);
      }
    `;
  }

  async priceOption(params: OptionParams): Promise<PricingResult> {
    const startTime = performance.now();

    if (!this.device || !this.computePipeline || !this.bindGroupLayout) {
      throw new Error('WebGPU not initialized');
    }

    // Create input buffer
    const paramsArray = new Float32Array([
      params.S,
      params.K,
      params.r,
      params.T,
      params.sigma,
      params.paths,
      params.steps,
      Math.random() * 1000000 // Random seed
    ]);

    const paramsBuffer = this.device.createBuffer({
      label: 'Parameters Buffer',
      size: paramsArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    new Float32Array(paramsBuffer.getMappedRange()).set(paramsArray);
    paramsBuffer.unmap();

    // Create output buffer
    const resultSize = params.paths * 4; // 4 bytes per float
    const resultsBuffer = this.device.createBuffer({
      label: 'Results Buffer',
      size: resultSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create staging buffer for reading results
    const stagingBuffer = this.device.createBuffer({
      label: 'Staging Buffer',
      size: resultSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      label: 'Monte Carlo Bind Group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: resultsBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(params.paths / 256));
    passEncoder.end();

    // Copy results to staging buffer
    commandEncoder.copyBufferToBuffer(
      resultsBuffer, 0,
      stagingBuffer, 0,
      resultSize
    );

    this.device.queue.submit([commandEncoder.finish()]);

    // Read results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    // Calculate statistics
    const mean = results.reduce((a, b) => a + b, 0) / params.paths;
    const variance = results.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (params.paths - 1);
    const standardError = Math.sqrt(variance / params.paths);
    const confidence95: [number, number] = [
      mean - 1.96 * standardError,
      mean + 1.96 * standardError
    ];

    const executionTime = performance.now() - startTime;

    // Cleanup
    paramsBuffer.destroy();
    resultsBuffer.destroy();
    stagingBuffer.destroy();

    return {
      price: mean,
      standardError,
      executionTime,
      pathsPerSecond: params.paths / (executionTime / 1000),
      confidence95,
      device: 'webgpu'
    };
  }

  async calculateGreeks(params: OptionParams): Promise<{
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
  }> {
    const epsilon = 0.01;

    // Calculate base price
    const basePrice = await this.priceOption(params);

    // Delta: ∂V/∂S
    const upS = await this.priceOption({ ...params, S: params.S * (1 + epsilon) });
    const downS = await this.priceOption({ ...params, S: params.S * (1 - epsilon) });
    const delta = (upS.price - downS.price) / (2 * params.S * epsilon);

    // Gamma: ∂²V/∂S²
    const gamma = (upS.price - 2 * basePrice.price + downS.price) /
                  Math.pow(params.S * epsilon, 2);

    // Theta: ∂V/∂T (negative because we measure decay)
    const thetaParam = { ...params, T: params.T * (1 - epsilon) };
    const thetaPrice = await this.priceOption(thetaParam);
    const theta = -(basePrice.price - thetaPrice.price) / (params.T * epsilon * 365);

    // Vega: ∂V/∂σ
    const upSigma = await this.priceOption({ ...params, sigma: params.sigma + epsilon });
    const vega = (upSigma.price - basePrice.price) / (epsilon * 100);

    // Rho: ∂V/∂r
    const upR = await this.priceOption({ ...params, r: params.r + epsilon });
    const rho = (upR.price - basePrice.price) / (epsilon * 100);

    return { delta, gamma, theta, vega, rho };
  }

  destroy() {
    if (this.device) {
      this.device.destroy();
    }
  }
}

// CPU fallback implementation
export class MonteCarloCPU {
  private normalRandom(): number {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  async priceOption(params: OptionParams): Promise<PricingResult> {
    const startTime = performance.now();
    const dt = params.T / params.steps;
    const drift = (params.r - 0.5 * params.sigma * params.sigma) * dt;
    const diffusion = params.sigma * Math.sqrt(dt);
    const discount = Math.exp(-params.r * params.T);

    const payoffs: number[] = [];

    for (let i = 0; i < params.paths; i++) {
      let price = params.S;

      for (let j = 0; j < params.steps; j++) {
        const z = this.normalRandom();
        price *= Math.exp(drift + diffusion * z);
      }

      const payoff = Math.max(price - params.K, 0);
      payoffs.push(payoff * discount);
    }

    const mean = payoffs.reduce((a, b) => a + b, 0) / params.paths;
    const variance = payoffs.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (params.paths - 1);
    const standardError = Math.sqrt(variance / params.paths);
    const confidence95: [number, number] = [
      mean - 1.96 * standardError,
      mean + 1.96 * standardError
    ];

    const executionTime = performance.now() - startTime;

    return {
      price: mean,
      standardError,
      executionTime,
      pathsPerSecond: params.paths / (executionTime / 1000),
      confidence95,
      device: 'cpu'
    };
  }
}