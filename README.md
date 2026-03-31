# QIP Paper I: Foundations and Gravity — Companion Code

Numerical verification scripts supporting:

> M. Travan, *Quantum Information Permeability: Foundations and Gravity* (Paper I)

## Scripts

| Script | What it verifies | Paper reference |
|--------|-----------------|-----------------|
| `p7_born_rule.py` | Born rule emergence from Haar measure on S^(2n-1) restricted to the QIP fibre. Analytical + Monte Carlo verification of Gleason's theorem conditions. | Appendix F (Born rule) |
| `qip_hartree_full.py` | Full density-matrix Hartree optimisation of the throughput functional J with octonionic (dim=8) jump operators and GKSL channel structure. Verifies MTP stationarity. | Appendix A (EFE derivation) |
| `e8_summary.py` | E8 root system (240 roots), Cartan matrix, F4 Cartan orthogonality to Y/B-L/T3, and lattice properties used in the continuum limit argument. | Appendix D (rho-dynamics, E8 continuum limit) |

## Requirements

- Python 3.8+
- NumPy
- SciPy

No other dependencies.

## Usage

Each script is self-contained and can be run directly:

```bash
python p7_born_rule.py
python qip_hartree_full.py
python e8_summary.py
```
