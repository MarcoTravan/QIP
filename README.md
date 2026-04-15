# QIP Paper I: Foundations and Gravity — Companion Code
 
Numerical verification scripts supporting:
 
> M. Travan, *Quantum Information Permeability: Foundations and Gravity* (Paper I)
 
## Scripts
 
| Script | What it verifies | Paper reference |
|--------|-----------------|-----------------|
| `p7_born_rule.py` | Born rule emergence from Haar measure on S^(2n-1) restricted to the QIP fibre. Analytical + Monte Carlo verification of Gleason's theorem conditions. | Appendix F (Born rule) |
| `qip_hartree_full.py` | Full density-matrix Hartree optimisation of the throughput functional J with octonionic (dim=8) jump operators and GKSL channel structure. Verifies MTP stationarity. | Appendix A (EFE derivation) |
| `e8_summary.py` | E8 root system (240 roots), Cartan matrix, F4 Cartan orthogonality to Y/B-L/T3, and lattice properties used in the continuum limit argument. | Appendix D (ρ-dynamics, E8 continuum limit) |
| `ricci_biconformal.py` | Full symbolic derivation (sympy) + numerical verification of the Ricci tensor for the biconformal metric. Confirms: trace-free Ricci identity, Laplacian cancellation in the trace-free projection, Ricci component formulae, and the null contraction identity. 20 random configurations, all errors < 10⁻¹⁵. | §6.2 (Full EFE from MTP), Appendix B (Trace-free Ricci) |
| `maxwell_fano_verify.py` | All 12 source-free Maxwell equations from the Fano algebra structure constants, with split equation (Faraday RIGHT, Ampère LEFT) and B_y = −φ₆. Also verifies: bicycle-wheel states for all three propagation directions, in-phase E/B projections, Poynting vector for both polarisations, constant null energy-momentum tensor T^μν = k^μk^ν, and (+,+,−) spatial cross-product pattern. No external dependencies. | §6.9 (Maxwell from Fano algebra), Appendix L (Full Maxwell derivation) |
 
## Requirements
 
- Python 3.8+
- NumPy
- SciPy
 
`ricci_biconformal.py` additionally requires SymPy.
 
`maxwell_fano_verify.py` has no dependencies beyond the Python standard library.
 
## Usage
 
Each script is self-contained and can be run directly:
 
```bash
python p7_born_rule.py
python qip_hartree_full.py
python e8_summary.py
python ricci_biconformal.py
python maxwell_fano_verify.py
```
