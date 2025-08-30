import type { MapFamily, SamplerParams } from "../engine/types";

export interface SharedState {
  seed: number;
  mapSeed: number;
  family: MapFamily;
  params: Partial<SamplerParams>;
}

export function stateToUrl(state: SharedState): string {
  const url = new URL(window.location.href);
  
  url.searchParams.set('seed', String(state.seed));
  url.searchParams.set('mapSeed', String(state.mapSeed));
  url.searchParams.set('family', state.family);
  
  if (state.params.K) url.searchParams.set('K', String(state.params.K));
  if (state.params.T) url.searchParams.set('T', String(state.params.T));
  if (state.params.mu) url.searchParams.set('mu', String(state.params.mu));
  if (state.params.sigma) url.searchParams.set('sigma', String(state.params.sigma));
  if (state.params.rho) url.searchParams.set('rho', String(state.params.rho));
  
  return url.toString();
}

export function urlToState(): Partial<SharedState> {
  const url = new URL(window.location.href);
  const state: Partial<SharedState> = {};
  
  const seed = url.searchParams.get('seed');
  if (seed) state.seed = parseInt(seed, 10);
  
  const mapSeed = url.searchParams.get('mapSeed');
  if (mapSeed) state.mapSeed = parseInt(mapSeed, 10);
  
  const family = url.searchParams.get('family') as MapFamily;
  if (family) state.family = family;
  
  const params: Partial<SamplerParams> = {};
  const K = url.searchParams.get('K');
  if (K) params.K = parseInt(K, 10);
  
  const T = url.searchParams.get('T');
  if (T) params.T = parseInt(T, 10);
  
  const mu = url.searchParams.get('mu');
  if (mu) params.mu = parseFloat(mu);
  
  const sigma = url.searchParams.get('sigma');
  if (sigma) params.sigma = parseFloat(sigma);
  
  const rho = url.searchParams.get('rho');
  if (rho) params.rho = parseFloat(rho);
  
  if (Object.keys(params).length > 0) {
    state.params = params;
  }
  
  return state;
}