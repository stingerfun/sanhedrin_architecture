# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 10:26:06 2026

@author: Shaul Sapielkin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from typing import Tuple, Dict

def generate_correlation_matrix(n_models: int, avg_rho: float) -> np.ndarray:
    """Generate correlation matrix with specified average pairwise correlation."""
    C = np.ones((n_models, n_models)) * avg_rho
    np.fill_diagonal(C, 1.0)
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvals(C))
    if min_eig < 0:
        C += (-min_eig + 0.01) * np.eye(n_models)
    return C

def generate_correlated_outputs(
    mu: np.ndarray, 
    C: np.ndarray, 
    n_samples: int
) -> np.ndarray:
    """Generate correlated model outputs via Cholesky decomposition."""
    L = cholesky(C, lower=True)
    z = np.random.randn(n_samples, len(mu))
    return mu + z @ L.T

def compute_ensemble_error(
    outputs: np.ndarray, 
    ground_truth: float, 
    M: int
) -> float:
    """Compute MSE for ensemble of M models."""
    ensemble_pred = np.mean(outputs[:, :M], axis=1)
    return np.mean((ensemble_pred - ground_truth) ** 2)

def compute_optimal_size(
    E: float,  # Epistemic uncertainty
    S: float,  # Social criticality
    rho: float,  # Average correlation
    sigma2: float = 1.0,
    c_inf: float = 1.0,
    mu: float = 0.05,
    nu: float = 0.1,
    sigma_trust2: float = 4.0,
    M_base: int = 3,
    k_E: float = 8.0,
    k_S: float = 6.0,
    M_max: int = 20
) -> Tuple[int, Dict]:
    """Compute optimal council size via grid search over loss function."""
    
    # Target size from epistemic and social factors
    M_target = M_base + int(k_E * E / (1 - rho)) + int(k_S * S * (1 + E))
    M_target = max(3, min(M_target, M_max))
    
    # Grid search near target
    M_range = range(max(1, M_target - 5), min(M_max + 1, M_target + 5))
    
    best_M = M_base
    best_loss = float('inf')
    loss_profile = {}
    
    for M in M_range:
        # Prediction loss (bias-variance-covariance decomposition)
        L_error = sigma2 / M + ((M - 1) / M) * rho * sigma2
        
        # Resource cost
        L_cost = M * c_inf + 0.1 * M * np.log(M + 1)
        
        # Trust calibration (Gaussian centered at M_target)
        L_trust = np.exp(-((M - M_target) ** 2) / (2 * sigma_trust2))
        
        # Total loss
        L_total = L_error + mu * L_cost - nu * L_trust
        
        loss_profile[M] = {
            'error': L_error,
            'cost': L_cost,
            'trust': L_trust,
            'total': L_total
        }
        
        if L_total < best_loss:
            best_loss = L_total
            best_M = M
    
    # Ensure odd number for majority voting
    if best_M % 2 == 0:
        best_M += 1
    
    return best_M, loss_profile

def run_simulation(
    E_values: np.ndarray,
    S_values: np.ndarray,
    rho_values: np.ndarray,
    n_trials: int = 1000,
    n_models: int = 20
) -> Dict:
    """Run full Monte Carlo simulation."""
    
    results = {
        'optimal_sizes': {},
        'marginal_benefits': {},
        'effectiveness': {}
    }
    
    for E in E_values:
        for S in S_values:
            for rho in rho_values:
                key = (E, S, rho)
                
                # Compute optimal size
                M_opt, _ = compute_optimal_size(E, S, rho)
                results['optimal_sizes'][key] = M_opt
                
                # Monte Carlo trials
                errors_by_M = {M: [] for M in range(1, n_models + 1)}
                
                for trial in range(n_trials):
                    # Generate synthetic scenario
                    mu = np.random.randn(n_models) * 0.1  # Small biases
                    C = generate_correlation_matrix(n_models, rho)
                    ground_truth = 0.0
                    
                    outputs = generate_correlated_outputs(
                        mu, C, n_samples=100
                    )
                    
                    # Compute errors for different ensemble sizes
                    for M in range(1, n_models + 1):
                        error = compute_ensemble_error(outputs, ground_truth, M)
                        errors_by_M[M].append(error)
                
                # Compute marginal benefits
                avg_errors = {M: np.mean(errors_by_M[M]) for M in errors_by_M}
                marginal = [
                    avg_errors[M] - avg_errors[M + 1] 
                    for M in range(1, n_models)
                ]
                results['marginal_benefits'][key] = marginal
                
                # Compute effectiveness thresholds
                max_reduction = avg_errors[1] - avg_errors[n_models]
                effectiveness = {}
                for eta in [0.9, 0.95, 0.99]:
                    target_error = avg_errors[n_models] + (1 - eta) * max_reduction
                    M_eta = next(
                        (M for M in range(1, n_models + 1) 
                         if avg_errors[M] <= target_error),
                        n_models
                    )
                    effectiveness[eta] = M_eta
                results['effectiveness'][key] = effectiveness
    
    return results

def plot_results(results: Dict) -> None:
    """Generate publication-quality plots."""
    # Figure 1: Optimal size vs parameters
    # Figure 2: Marginal benefit decay
    # Figure 3: Effectiveness thresholds
    # (Implementation details omitted for brevity)
    pass

# Main execution
if __name__ == "__main__":
    E_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    S_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    rho_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    print("Running Monte Carlo simulation...")
    results = run_simulation(E_values, S_values, rho_values, n_trials=1000)
    
    print(f"Simulation complete. Sample result:")
    sample_key = (0.5, 0.6, 0.3)
    print(f"E={sample_key[0]}, S={sample_key[1]}, ρ={sample_key[2]}")
    print(f"Optimal council size: {results['optimal_sizes'][sample_key]}")
    print(f"Effectiveness M_0.9: {results['effectiveness'][sample_key][0.9]}")

