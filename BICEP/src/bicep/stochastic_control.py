import os
import numpy as _np
import logging
from functools import lru_cache

# ——— Logging setup ———
LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,  # backends/
    os.pardir,  # project root
    "results", "logs", "stochastic_control.log"
)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ——— Tunable thresholds ———
HIGH_THRESHOLD = 10
LOW_THRESHOLD = 2
DECAY_RATE = 0.1
CONTROL_WARNING_THRESHOLD = 0.9
ADAPTIVE_ADJUSTMENT = 0.05
MAX_HIGH_THRESHOLD = 100

def _log(msg):
    logging.info(msg)

def adjust_variance(base_variance, adjustment_factor=1.0):
    """
    Adjust the base variance by a scaling factor.
    
    Args:
        base_variance: Base variance value
        adjustment_factor: Multiplicative adjustment factor
        
    Returns:
        Adjusted variance value
    """
    adj = base_variance * adjustment_factor
    _log(f"Adjusted variance {base_variance}→{adj} (factor={adjustment_factor})")
    return adj

def adaptive_randomness_control(brownian_increment, control_parameter, target_range=(0.2, 1.0)):
    """
    Apply adaptive control to stochastic increments.
    
    Args:
        brownian_increment: Random increment value
        control_parameter: Control strength parameter [0, 1]
        target_range: Allowed range for scaling factor
        
    Returns:
        Controlled increment
    """
    if control_parameter >= CONTROL_WARNING_THRESHOLD:
        logging.warning(f"High control parameter: {control_parameter}")
    scale = _np.clip(0.5 + control_parameter * 0.5, *target_range)
    adj = brownian_increment * scale
    _log(f"Control adjustment: {brownian_increment}→{adj} (scale={scale})")
    return adj

def control_randomness_by_state(state_visit_count, total_steps,
                                high_threshold=HIGH_THRESHOLD,
                                low_threshold=LOW_THRESHOLD):
    """
    Adjust randomness based on state visitation statistics.
    
    Args:
        state_visit_count: Number of visits to current state
        total_steps: Total number of simulation steps
        high_threshold: Upper threshold for state visits
        low_threshold: Lower threshold for state visits
        
    Returns:
        Variance scaling factor
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    norm = state_visit_count / total_steps
    global HIGH_THRESHOLD
    if HIGH_THRESHOLD < MAX_HIGH_THRESHOLD and norm > 0.8 * high_threshold:
        HIGH_THRESHOLD = min(MAX_HIGH_THRESHOLD, HIGH_THRESHOLD + ADAPTIVE_ADJUSTMENT)
        _log(f"Bumped HIGH_THRESHOLD → {HIGH_THRESHOLD}")
    if norm < low_threshold:
        factor = 1.5
    elif norm > high_threshold:
        factor = 0.5
    else:
        factor = 1.0
    _log(f"State control: count={state_visit_count}/{total_steps} → factor={factor}")
    return factor

def combined_variance_control(state_visit_count, total_steps, t,
                              base_variance=1.0,
                              decay_rate=DECAY_RATE):
    """
    Combine state-based and time-based variance controls.
    
    Args:
        state_visit_count: Number of state visits
        total_steps: Total simulation steps
        t: Current time
        base_variance: Base variance level
        decay_rate: Temporal decay rate
        
    Returns:
        Combined variance scaling factor
    """
    sf = control_randomness_by_state(state_visit_count, total_steps)
    tf = _np.exp(-decay_rate * t)
    comb = base_variance * sf * tf
    _log(f"Combined variance @ t={t}: {comb}")
    return comb

def apply_stochastic_controls(brownian_increment,
                              state_visit_count,
                              control_parameter,
                              t,
                              total_steps,
                              base_variance=1.0):
    """
    Apply comprehensive stochastic controls to increment.
    
    This function combines multiple control mechanisms:
    - State-based variance adjustment
    - Time-dependent decay
    - Parameter-based scaling
    
    Args:
        brownian_increment: Base random increment
        state_visit_count: Current state visit count
        control_parameter: Process control parameter [0, 1]
        t: Current time
        total_steps: Total simulation steps
        base_variance: Base variance level
        
    Returns:
        Controlled stochastic increment
    """
    vf = combined_variance_control(state_visit_count, total_steps, t, base_variance)
    ff = adaptive_randomness_control(brownian_increment, control_parameter)
    final = ff * vf
    _log(f"Final increment: {final}")
    return final
