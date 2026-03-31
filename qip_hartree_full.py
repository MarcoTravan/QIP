#!/usr/bin/env python3
"""
QIP Full Density-Matrix Hartree with Octonionic Jump Operators

The throughput per channel uses the GKSL jump operators L_a (a=1..7),
which are the 8×8 matrices representing left-multiplication by the
imaginary octonion units e_a. The channel a throughput between nodes:

  F_a(i,j) = Tr(L_a ρ̂_i L_a† ρ̂_j)

Since L_a is anti-Hermitian (L_a† = -L_a), this equals:
  F_a(i,j) = -Tr(L_a ρ̂_i L_a ρ̂_j) = Tr((L_a†ρ̂_i†L_a) ρ̂_j)
           = Tr((L_a ρ̂_i L_a) ρ̂_j)  [since ρ̂ Hermitian, L_a anti-Herm]

Actually more carefully: L_a† = -L_a (anti-symmetric real matrix).
  F_a = Tr(L_a ρ̂_i L_a^T ρ̂_j) = Tr(L_a ρ̂_i (-L_a) ρ̂_j)

For the throughput to be POSITIVE: use |F_a| or use L_a ρ̂ L_a^T
(the GKSL Kraus form, which gives a positive channel).

The GKSL channel: ℰ_a(ρ̂) = L_a ρ̂ L_a^T (for real anti-symmetric L_a)
Throughput: Tr(ℰ_a(ρ̂_i) ρ̂_j) = Tr(L_a ρ̂_i L_a^T ρ̂_j)

Each spatial edge couples to 2 channels (one Fano pair = 2 octonion units).
The edge along spatial direction d couples to channels in pair d,
weighted by the edge's component in that direction.
The internal channel e₇ couples to ALL edges equally.

Usage:
    python qip_hartree_full.py --R 2 --trials 30 --rho_avg 0.2 --verbose
    python qip_hartree_full.py --R 2 3 --trials 20 --rho_avg 0.2
"""

import os
import numpy as np
from math import pi, sqrt, log
from collections import defaultdict
import argparse
import json
import time

import torch

# GPU setup
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}")
else:
    DEVICE = torch.device('cpu')
    print(f"CPU mode, PyTorch {torch.__version__}")

DTYPE = torch.float64

# ================================================================
# QIP CONSTANTS
# ================================================================
ALPHA = 2 * pi - 0.5
D_BATH = 8
RHO_CRIT = 1 - 1/D_BATH

# Fano pairs: spatial direction d → octonion units (a, b)
# Pair 0 (x-direction): e₄, e₅
# Pair 1 (y-direction): e₂, e₆  (note: {e₆,e₂} reordered for consistency)
# Pair 2 (z-direction): e₁, e₃
FANO_PAIRS = [(4, 5), (2, 6), (1, 3)]
FANO_INTERNAL = 7  # e₇


# ================================================================
# OCTONION JUMP OPERATORS
# ================================================================

def build_octonion_L():
    """Build the 7 jump operator matrices L_a (a=1..7).
    
    L_a is the 8×8 real matrix for left-multiplication by e_a.
    (e_a × e_j) = Σ_i (L_a)_{ij} e_i
    
    L_a is anti-symmetric: L_a^T = -L_a.
    L_a² = -I for all a=1..7.
    """
    # Fano lines: ordered triples (a,b,c) with e_a × e_b = +e_c
    fano_lines = [
        (1,2,3), (1,4,5), (1,7,6),
        (2,4,6), (2,5,7),
        (3,4,7), (3,6,5)
    ]
    
    # Structure constants
    epsilon = {}
    for a in range(8):
        for b in range(8):
            epsilon[(a,b)] = (0, 0)
    
    for a in range(8):
        epsilon[(0,a)] = (1, a)
        epsilon[(a,0)] = (1, a)
    
    for a in range(1, 8):
        epsilon[(a,a)] = (-1, 0)
    
    for (a,b,c) in fano_lines:
        epsilon[(a,b)] = (1, c)
        epsilon[(b,a)] = (-1, c)
        epsilon[(b,c)] = (1, a)
        epsilon[(c,b)] = (-1, a)
        epsilon[(c,a)] = (1, b)
        epsilon[(a,c)] = (-1, b)
    
    # Build matrices
    L = np.zeros((8, 8, 8))
    for a in range(8):
        for j in range(8):
            sign, idx = epsilon[(a,j)]
            if sign != 0:
                L[a, idx, j] = sign
    
    return L  # L[a] is the 8×8 matrix for e_a


# ================================================================
# LATTICE
# ================================================================

def build_bcc_ball(R):
    nodes = []
    imax = int(R) + 1
    for i in range(-imax, imax+1):
        for j in range(-imax, imax+1):
            for k in range(-imax, imax+1):
                if i*i + j*j + k*k <= R*R:
                    nodes.append((float(i), float(j), float(k)))
                x, y, z = i+0.5, j+0.5, k+0.5
                if x*x + y*y + z*z <= R*R:
                    nodes.append((x, y, z))
    return nodes

def build_edges(nodes):
    node_set = set(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = []
    for i, n in enumerate(nodes):
        for sx in [-0.5, 0.5]:
            for sy in [-0.5, 0.5]:
                for sz in [-0.5, 0.5]:
                    nb = (round(n[0]+sx,1), round(n[1]+sy,1), round(n[2]+sz,1))
                    if nb in node_set:
                        j = node_idx[nb]
                        if i < j:
                            edges.append((i,j))
    return edges


# ================================================================
# THROUGHPUT ENGINE WITH JUMP OPERATORS
# ================================================================

class JumpOperatorEngine:
    """Throughput using octonionic jump operators L_a.
    
    J = Σ_edges κ(ρ_i,ρ_j) × Σ_a w_a(edge) × Tr(L_a ρ̂_i L_a^T ρ̂_j)
    
    where the sum over a runs over the 7 imaginary octonion units,
    and w_a(edge) is the weight of channel a for this edge:
      - Channels in pair d (2 per pair): weighted by edge direction w_d
      - Channel 7 (internal): equal weight for all edges (1/7)
    """
    
    def __init__(self, nodes, edges, d=D_BATH):
        self.N = len(nodes)
        self.d = d
        self.nodes = nodes
        self.n_edges = len(edges)
        
        ea = np.array(edges, dtype=np.int64)
        self.ei = torch.tensor(ea[:, 0], device=DEVICE, dtype=torch.long)
        self.ej = torch.tensor(ea[:, 1], device=DEVICE, dtype=torch.long)
        
        # Edge spatial direction weights
        dirs = np.zeros((len(edges), 3))
        for e, (i,j) in enumerate(edges):
            for dd in range(3):
                dirs[e, dd] = nodes[j][dd] - nodes[i][dd]
        nsq = np.maximum(np.sum(dirs**2, axis=1, keepdims=True), 1e-10)
        spatial_w = dirs**2 / nsq  # (n_edges, 3)
        
        # Build jump operator tensors on device
        L_np = build_octonion_L()
        
        # For each of the 7 channels, compute the weight per edge
        # Channel a belongs to pair d if a ∈ FANO_PAIRS[d]
        # Channel 7 is internal
        channel_to_pair = {}
        for d, (a, b) in enumerate(FANO_PAIRS):
            channel_to_pair[a] = d
            channel_to_pair[b] = d
        channel_to_pair[FANO_INTERNAL] = -1  # internal
        
        # Build per-edge weights for each channel: shape (n_edges, 7)
        # Channel a (a=1..7) → index a-1 in the array
        channel_weights = np.zeros((len(edges), 7))
        for a in range(1, 8):
            idx = a - 1
            pair = channel_to_pair[a]
            if pair >= 0:
                # Spatial channel: weight = (1/7) × 3 × spatial_w[pair_d]
                # Normalisation: 7 channels total, 2 per pair × 3 pairs + 1 internal = 7
                # Each channel gets base weight 1/7.
                # Spatial modulation: channel in pair d gets extra weight from edge direction d
                # w_a = (1/7) × (1 + 2 × spatial_w[d])  ... no, let me think clearly.
                #
                # Simple model: the throughput through channel a is proportional to
                # how much the edge "points" in the direction of pair d containing a.
                # w_a(edge) = spatial_w[d(a)] for spatial channels
                # w_7(edge) = 1/3 (equal for all edges on BCC where spatial_w sums to 1)
                # Then normalise so total weight per edge = 1:
                # Σ_a w_a = Σ_{d=0,1,2} 2×spatial_w[d] + 1×(1/3) = 2×1 + 1/3 = 7/3
                # Normalised: w_a → w_a × 3/7
                channel_weights[:, idx] = spatial_w[:, pair] * 3.0 / 7.0
            else:
                # Internal channel: w_7 = (1/3) × 3/7 = 1/7
                channel_weights[:, idx] = 1.0 / 7.0
        
        self.channel_weights = torch.tensor(channel_weights, device=DEVICE, dtype=DTYPE)
        
        # Jump operator matrices on device: (7, d, d) for channels 1..7
        L_tensors = torch.tensor(L_np[1:8], device=DEVICE, dtype=DTYPE)  # (7, 8, 8)
        self.L = L_tensors       # L[c] = L_{c+1} for c=0..6
        self.Lt = L_tensors.transpose(1, 2)  # L^T (= -L for anti-symmetric)
    
    def compute_J(self, all_params):
        """Compute throughput with jump operators.
        
        J = Σ_edges κ × Σ_{c=0..6} w_c(edge) × Tr(L_c ρ̂_i L_c^T ρ̂_j)
        """
        N = self.N; d = self.d
        
        # Density matrices
        V = all_params.reshape(N, d, d)
        VVt = torch.bmm(V, V.transpose(1, 2))
        traces = torch.diagonal(VVt, dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(2)
        rho_mats = VVt / traces
        
        # Congestion
        rho_sq = torch.bmm(rho_mats, rho_mats)
        tr_rho_sq = torch.diagonal(rho_sq, dim1=1, dim2=2).sum(dim=1)
        congestion = 1.0 - tr_rho_sq
        
        # Permeability
        ci = congestion[self.ei]
        cj = congestion[self.ej]
        kappa = 1.0 / (1.0 + ALPHA * (ci + cj))
        
        # Edge density matrices
        rho_i = rho_mats[self.ei]  # (n_edges, d, d)
        rho_j = rho_mats[self.ej]  # (n_edges, d, d)
        
        # Channel throughputs
        fidelity = torch.zeros(self.n_edges, device=DEVICE, dtype=DTYPE)
        
        for c in range(7):
            Lc = self.L[c]    # (d, d)
            Lct = self.Lt[c]  # (d, d) = L_c^T
            
            # L_c ρ̂_i L_c^T for all edges: (n_edges, d, d)
            # = Lc @ rho_i @ Lct (batch)
            Lc_rho_i = torch.matmul(Lc, rho_i)        # (n_edges, d, d)
            Lc_rho_i_Lct = torch.matmul(Lc_rho_i, Lct)  # (n_edges, d, d)
            
            # F_c = Tr(L_c ρ̂_i L_c^T × ρ̂_j)
            F_c = torch.diagonal(
                torch.bmm(Lc_rho_i_Lct, rho_j), dim1=1, dim2=2
            ).sum(dim=1)  # (n_edges,)
            
            # Weighted by channel weight for this edge
            w_c = self.channel_weights[:, c]  # (n_edges,)
            fidelity = fidelity + w_c * F_c
        
        J = torch.sum(kappa * fidelity)
        return J, congestion, rho_mats
    
    def measure_z3(self, rho_mats):
        """Measure ℤ₃ breaking from density matrix structure."""
        d = self.d; N = self.N
        
        diags = torch.diagonal(rho_mats, dim1=1, dim2=2)  # (N, d)
        
        pair_weights = torch.zeros(N, 3, device=DEVICE, dtype=DTYPE)
        for p, (a, b) in enumerate(FANO_PAIRS):
            pair_weights[:, p] = diags[:, a] + diags[:, b]
        
        real_weight = diags[:, 0]
        internal_weight = diags[:, FANO_INTERNAL]
        
        avg_pair = pair_weights.mean(dim=0)
        ps = torch.sort(avg_pair, descending=True).values
        aniso = float(ps[0] / ps[2]) if float(ps[2]) > 1e-10 else 1.0
        
        # Also measure the CHANNEL throughput anisotropy
        # (how much throughput goes through each pair's channels)
        # This requires the full density matrices, computed per channel.
        
        return {
            'pair_weights': avg_pair.detach().cpu().numpy().tolist(),
            'pair_1': float(avg_pair[0]),
            'pair_2': float(avg_pair[1]),
            'pair_3': float(avg_pair[2]),
            'anisotropy': aniso,
            'real_weight': float(real_weight.mean()),
            'internal_weight': float(internal_weight.mean()),
        }


# ================================================================
# INITIAL CONDITIONS
# ================================================================

def make_initial_params(N, d, rho_avg, mode='random', seed=42):
    torch.manual_seed(seed)
    
    if mode == 'random':
        mix = sqrt(rho_avg * d / (d-1))
        params = torch.eye(d, device=DEVICE, dtype=DTYPE).unsqueeze(0).repeat(N, 1, 1)
        params = params + torch.randn(N, d, d, device=DEVICE, dtype=DTYPE) * mix * 0.3
    elif mode.startswith('z3_'):
        axis = {'z3_x': 0, 'z3_y': 1, 'z3_z': 2}[mode]
        pair = FANO_PAIRS[axis]
        params = torch.eye(d, device=DEVICE, dtype=DTYPE).unsqueeze(0).repeat(N, 1, 1)
        for idx in pair:
            params[:, idx, idx] *= 2.0
        params = params + torch.randn(N, d, d, device=DEVICE, dtype=DTYPE) * 0.2
    elif mode == 'peaked':
        params = torch.zeros(N, d, d, device=DEVICE, dtype=DTYPE)
        params[:, 0, 0] = 3.0
        params = params + torch.randn(N, d, d, device=DEVICE, dtype=DTYPE) * 0.1
    elif mode == 'mixed':
        # Start with significant mixing across all directions
        params = torch.randn(N, d, d, device=DEVICE, dtype=DTYPE) * 0.5
        params = params + torch.eye(d, device=DEVICE, dtype=DTYPE).unsqueeze(0) * 0.3
    else:
        params = torch.randn(N, d, d, device=DEVICE, dtype=DTYPE)
    
    return params.reshape(N, d*d)


# ================================================================
# OPTIMISATION
# ================================================================

def optimise(nodes, edges, rho_avg, n_trials=20, maxiter=2000, verbose=False):
    N = len(nodes)
    d = D_BATH
    
    eng = JumpOperatorEngine(nodes, edges, d)
    
    # Uniform baseline
    uniform_p = torch.eye(d, device=DEVICE, dtype=DTYPE).unsqueeze(0).repeat(N, 1, 1).reshape(N, d*d)
    J_uni, _, _ = eng.compute_J(uniform_p)
    J_uni = float(J_uni)
    
    best_J = -1e10
    best_params = None
    
    modes = ['random', 'z3_x', 'z3_y', 'z3_z', 'peaked', 'mixed']
    
    target_rho = N * rho_avg
    penalty_lambda = 1000.0
    
    for trial in range(n_trials):
        mode = modes[trial % len(modes)]
        seed = 42 + trial
        
        params = make_initial_params(N, d, rho_avg, mode, seed)
        params = params.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.LBFGS(
            [params], lr=1.0, max_iter=maxiter,
            tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=20, line_search_fn='strong_wolfe'
        )
        
        step_count = [0]
        
        def closure():
            optimizer.zero_grad()
            J, congestion, _ = eng.compute_J(params)
            total_rho = torch.sum(congestion)
            penalty = penalty_lambda * (total_rho - target_rho)**2
            neg_pen = 1000.0 * torch.sum(torch.clamp(-congestion, min=0)**2)
            loss = -J + penalty + neg_pen
            loss.backward()
            step_count[0] += 1
            return loss
        
        try:
            optimizer.step(closure)
            
            with torch.no_grad():
                J_final, cong, rho_mats = eng.compute_J(params)
                J_val = float(J_final)
                
                if J_val > best_J:
                    best_J = J_val
                    best_params = params.detach().clone()
                    
                    if verbose:
                        z3_tmp = eng.measure_z3(rho_mats)
                        print(f"  Trial {trial} ({mode}): J={J_val:.4f}, "
                              f"aniso={z3_tmp['anisotropy']:.4f}, "
                              f"ρ_mean={float(cong.mean()):.4f}, "
                              f"steps={step_count[0]}")
        except Exception as e:
            if verbose:
                print(f"  Trial {trial} ({mode}): {e}")
    
    # Final analysis
    with torch.no_grad():
        J_final, cong, rho_mats = eng.compute_J(best_params)
        z3 = eng.measure_z3(rho_mats)
        cong_np = cong.cpu().numpy()
    
    return {
        'rho_avg_target': rho_avg,
        'J_uniform': J_uni,
        'J_best': best_J,
        'gain_pct': (best_J/J_uni - 1)*100 if J_uni > 0 else 0,
        'z3_breaking': z3,
        'congestion_stats': {
            'min': float(cong_np.min()),
            'max': float(cong_np.max()),
            'mean': float(cong_np.mean()),
            'std': float(cong_np.std()),
        },
    }


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='QIP Hartree — Jump Operators')
    parser.add_argument('--R', nargs='+', type=int, default=[2])
    parser.add_argument('--rho_avg', type=float, default=0.2)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"\nQIP Hartree — Octonionic Jump Operators")
    print(f"Device: {DEVICE}")
    print(f"α={ALPHA:.4f}, d={D_BATH}, ρ_crit={RHO_CRIT:.4f}")
    print(f"Fano pairs: {FANO_PAIRS}, internal: e_{FANO_INTERNAL}")
    print()
    
    all_results = {}
    
    for R in args.R:
        nodes = build_bcc_ball(R)
        edges = build_edges(nodes)
        N = len(nodes)
        
        print(f"{'='*65}")
        print(f"R={R}: N={N}, edges={len(edges)}, params={N*D_BATH**2}")
        print(f"{'='*65}")
        
        t0 = time.time()
        result = optimise(nodes, edges, args.rho_avg,
                         args.trials, args.maxiter, args.verbose)
        dt = time.time() - t0
        
        result['R'] = R; result['N'] = N
        result['n_edges'] = len(edges)
        result['time_sec'] = dt
        
        z3 = result['z3_breaking']
        cs = result['congestion_stats']
        
        print(f"\n  J_best = {result['J_best']:.4f} (uniform: {result['J_uniform']:.4f})")
        print(f"  Congestion: mean={cs['mean']:.4f}, std={cs['std']:.3f}")
        print(f"\n  ℤ₃ BREAKING:")
        print(f"    Pair 1 (e₄,e₅): {z3['pair_1']:.6f}")
        print(f"    Pair 2 (e₂,e₆): {z3['pair_2']:.6f}")
        print(f"    Pair 3 (e₁,e₃): {z3['pair_3']:.6f}")
        print(f"    Anisotropy: {z3['anisotropy']:.6f}")
        print(f"    Real (e₀): {z3['real_weight']:.6f}")
        print(f"    Internal (e₇): {z3['internal_weight']:.6f}")
        print(f"  Time: {dt:.1f}s")
        
        if z3['anisotropy'] > 1.01:
            print(f"\n  *** ℤ₃ BREAKING DETECTED ***")
        
        all_results[str(R)] = result
        print()
    
    outfile = os.path.join(args.outdir, 'full_dm_results.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {outfile}")


if __name__ == '__main__':
    main()
