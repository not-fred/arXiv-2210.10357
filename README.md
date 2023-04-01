# Script used in: Dynamic-based Entanglement Witnesses for Non-Gaussian States of Harmonic Oscillators [arXiv-2210.10357]
Script used in [arXiv:2210.10357](https://arxiv.org/abs/2210.10357) for minimizing
entanglement over positive partial transpose states, to find the threshold for a
dynamic-based entanglement witness. See the paper for more detail.

# Usage
    julia SDP-min-entanglement.jl 𝐧 θ P₃ [file]

## Arguments
- `𝐧::Integer`: Hilbert space truncation; energy constraints on the normal modes
- `θ::Float`: Angle, in radians, of passive transformation that relates physical
  modes to normal modes
- `P₃::Vector{Float}`: Observed violation of precession protocol
- `file::String`: File to save the data. Returns to standard output if none specified

# Package versions
For the reported results, version 1.6.6 of `julia` was used, with the packages
- `Convex`: 0.15.1
- `SCS`: 1.1.2
- `Memoization`: 0.1.14
