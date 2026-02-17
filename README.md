# Sanhedrin Architecture: Context-Aware Adaptive Model Council

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for the paper:

**"Sanhedrin Architecture: Context-Aware Adaptive Model Council for AI Decision Systems"**  
*Shaul Sapielkin* (2026)

## Overview

The Sanhedrin Architecture is a novel framework for determining the optimal number of AI models to participate in collective decisions. Unlike fixed ensembles or simple routing systems, it dynamically adapts council size based on:

- **Epistemic Uncertainty** — How well-established is the knowledge domain?
- **Social Criticality** — How high are the stakes of this decision?
- **Computational Constraints** — What resources are available?

Our Monte Carlo simulations across 125 parameter configurations show that optimal council sizes range from **3 to 15 models** in practice, with clear scaling laws for epistemic uncertainty and decision criticality.

## Key Results

| Configuration | Optimal Council Size |
|--------------|---------------------|
| Low stakes, established knowledge | 3-5 models |
| Medium stakes, contested knowledge | 7-11 models |
| High stakes, uncertain knowledge | 13-15 models |

Empirical effectiveness thresholds:
- **90% variance reduction**: 7.2 ± 0.4 models (theory: 9)
- **95% variance reduction**: 11.0 ± 0.2 models (theory: 19)
- **99% variance reduction**: 17.1 ± 0.4 models (theory: 99)

Real systems require **~20% fewer models** than worst-case theoretical predictions!

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Usage

Run the complete Monte Carlo simulation:

```bash
python sanhedrin_simulation.py
```

This will:
1. Simulate 125 parameter configurations (5×5×5 grid)
2. Run 1,000 Monte Carlo trials per configuration
3. Output validation statistics
4. Generate `results_summary.csv` with full results

**Expected runtime**: ~15 minutes on a standard laptop

### Example Output

```
Running Monte Carlo simulation...
Total configurations: 125
============================================================
Completed 25/125 configurations...
Completed 50/125 configurations...
...
Simulation complete!

=== VALIDATION RESULTS ===

1. CRITICALITY AMPLIFICATION (E=0.5, ρ=0.3):
   S=0.2 → M*=5
   S=0.4 → M*=7
   S=0.6 → M*=9
   S=0.8 → M*=11
   S=1.0 → M*=13

2. EFFECTIVENESS BOUNDS:
   η=0.90: M = 7.2 ± 0.4 (theory: 9)
   η=0.95: M = 11.0 ± 0.2 (theory: 19)
   η=0.99: M = 17.1 ± 0.4 (theory: 99)

3. OPTIMAL COUNCIL SIZE DISTRIBUTION:
   Mean: 10.7
   Median: 11
   Range: [3, 15]
```

## Mathematical Framework

The optimal council size minimizes:

```
L(M, q, c) = L_error(M) + μ·L_cost(M) - ν·L_trust(M)
```

Where:
- **L_error**: Prediction loss (bias² + variance/M + correlation penalty)
- **L_cost**: Computational cost (inference + synthesis)
- **L_trust**: Trust calibration (alignment with expected rigor)

### Closed-Form Solution

For the simplified accuracy-cost tradeoff:

```
M* = √[σ²(1-ρ̄) / (μ·c_inf)]
```

See the paper for full derivations and convergence proofs.

## Reproducibility

All results are fully reproducible with `np.random.seed(42)`. The simulation parameters are:

```python
E_values = [0.1, 0.3, 0.5, 0.7, 0.9]      # Epistemic uncertainty
S_values = [0.2, 0.4, 0.6, 0.8, 1.0]      # Social criticality
rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]    # Inter-model correlation
n_trials = 1000                             # Monte Carlo samples
```

## Citation

If you use this work, please cite:

```bibtex
@article{yourname2026sanhedrin,
  title={Sanhedrin Architecture: Context-Aware Adaptive Model Council for AI Decision Systems},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration: your.email@example.com

---

**Note**: This is a research prototype. Production deployment requires additional engineering, safety testing, and domain-specific calibration.

