# Twisted Eguchi-Kawai (TEK) Master Field — Reference

## Model

The Twisted Eguchi-Kawai (TEK) reduction of D-dimensional SU(N) lattice Yang-Mills
theory replaces the infinite-volume lattice by a single site with D unitary link
matrices. At N = ∞ the reduced model is equivalent to the infinite-volume theory
by volume independence (Eguchi-Kawai 1982; González-Arroyo-Okawa 1983).

Action:

    S_TEK = −(N/λ) Σ_{μ<ν} Re (1/N) Tr(z_μν · U_μ U_ν U_μ† U_ν†)

where:
- U_μ ∈ U(N) for μ = 1, …, D
- λ = g²N is the 't Hooft coupling, held fixed as N → ∞
- z_μν = exp(2πi n_μν / N) is the twist phase; n_μν is an antisymmetric integer tensor
- the path integral weight is exp(−S)

## Why the Twist

The untwisted Eguchi-Kawai model (z_μν = 1) breaks its Z_N^D center symmetry in
the weak-coupling regime, invalidating volume independence. The twist z_μν
creates an obstruction: the configurations minimizing the action are NOT fully
commuting U_μ (as they would be for untwisted EK), but rather clock-and-shift
matrices satisfying the Heisenberg relation U_μ U_ν = z_μν U_ν U_μ. This
obstruction pins center symmetry and restores volume independence.

## Symmetric Twist

Choose N = L² with L prime. The symmetric flux is:

    n_μν = k · L · ε_μν

where ε is an antisymmetric integer matrix with det = 1 (mod L), and k is the
flux integer. The explicit choices are:

- **D = 2**: only (1,2) plane is twisted. n₁₂ = kL, so z₁₂ = exp(2πi k/L).
  Safe for any k coprime to L.
- **D = 3**: one twisted plane, usually (1,2). n₁₂ = kL, others zero. Safe.
- **D = 4**: two twisted planes, (1,2) and (3,4). n₁₂ = n₃₄ = kL, others zero.
  **Caveat — see next section.**

Allowed matrix sizes: N ∈ {4, 9, 25, 49, 121, 169, 289, 361, 529, 841, …}
(squares of primes L = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, …).

## Center-Symmetry Breaking in D = 4 (Caveat R1)

For D = 4 with the symmetric flux k = 1, center symmetry is spontaneously broken
at sufficiently large N (≥ 100) across a wide coupling range. This was observed
in hep-th/0612097 (Teper et al. 2006) via Polyakov loop measurements. The broken
phase is associated with "fluxon" configurations that make the large-N reduction
fail.

González-Arroyo and Okawa (arXiv:1005.1981, 2010) showed that k = k_opt with

    k_opt ≈ L / 2   (rounded, coprime to L)

restores center symmetry at all couplings and is the recommended choice for D = 4.
Explicitly: L = 7 → k = 3; L = 11 → k = 5; L = 13 → k = 5 or 6; L = 17 → k = 8.

For D = 2 and D = 3 the symmetric flux k = 1 is safe.

## Master Field Ansatz (Center-Symmetric Orientation)

The center-symmetric saddle has U_μ with eigenvalues locked to the N-th roots of
unity. Parametrize:

    U_μ = Ω_μ · Γ · Ω_μ†,   Γ = diag(1, ω, ω², …, ω^{N-1}),   ω = exp(2πi/N)

Gauge-fix Ω_1 = I (so U_1 = Γ). The unknowns are Ω_2, …, Ω_D, each parametrized
by a Hermitian generator:

    Ω_μ = exp(i H_μ),   H_μ = H_μ†

giving (D − 1) · N² real parameters. The Haar measure on the conjugation orbit
{g Γ g†} is proportional to the Vandermonde of the eigenvalues, which is CONSTANT
since the eigenvalues are fixed. In this ansatz the effective action is the
classical TEK action itself.

**Saddle characterization.** The extremum of S_TEK on this submanifold is the
configuration that maximizes Σ Re[z_μν · Tr(U_μ U_ν U_μ† U_ν†)] / N. In the
twisted case this is achieved at the Heisenberg clock-and-shift pair:

    U_1 = Γ,   U_2 = S,   with   Γ S = z₁₂ · S Γ

and the corresponding plaquette value is W[□] = 1. This is the CLASSICAL TEK
vacuum. Whether it coincides with the quantum master field at a given coupling
is a coupling- and ansatz-dependent question (see "Open Question" below).

## Rectangular Wilson Loop — Twist Phase (R2, GATED)

For a rectangle of size R × T in the (μ, ν) plane on the single-site TEK lattice,
the Wilson loop picks up a twist phase from the Heisenberg non-commutativity:

    W[R × T]_{μν} = Tr[ f(R, T, z_μν) · U_μ^R · U_ν^T · U_μ^{-R} · U_ν^{-T} ] / N

The correct phase f(R, T, z) is NOT simply z^{R·T}. It is derived in:

- González-Arroyo & Okawa, Phys. Rev. D 27 (1983) 2397, eq. (3.5).
- García Pérez, González-Arroyo, Okawa, arXiv:1708.00841 (2017) — Wilson loops
  to order g⁴ on twisted lattices and reduced models.

**Status**: Not yet transcribed in this repo. `observables.wilson_loop_rectangular`
raises `NotImplementedError` until the formula is copied in (and tested against
the strong-coupling product law `W[R×T] ≈ W[□]^{RT}` as a partial check).

## Open Question: Classical vs Quantum Saddle

The ansatz above uses a Haar measure that, restricted to the orientation
manifold, is constant. The extremum of the classical action is the saddle of
that restricted measure. This gives the "classical TEK solution" — plaquette = 1
(or exp(2πi/L)-dependent at finite L) independent of λ.

The PHYSICAL master field at N = ∞ is the saddle of the full effective action
(classical + fluctuation / entropy). For lattice YM this has a coupling-dependent
plaquette that approaches the classical saddle only in the weak-coupling limit.

Three possible outcomes for direct optimization of S_TEK on the orientation
submanifold:

1. **Best case:** the ansatz captures the master field at all couplings because
   the measure is genuinely flat. (This is what the Gopakumar-Gross "master
   field in axial gauge" result for 2D suggests.) Plaquette matches MC at every λ.
2. **Partial:** ansatz captures the weak-coupling master field (where classical
   saddle dominates) but not strong coupling. Plaquette stays near 1 regardless
   of λ at strong coupling.
3. **Failure:** the classical saddle and master field differ significantly.
   In this case we must enlarge the ansatz (e.g., relax the exact-roots-of-unity
   constraint to allow eigenvalue fluctuations → R4 fallback: full
   U_μ = exp(i M_μ) with M_μ Hermitian, D · N² real parameters).

Phase B (untwisted EK, D = 2) and Phase C (TEK, D = 2) will discriminate between
these outcomes against known MC data.

## Verification Criteria

- **Unitarity:** ||U_μ U_μ† − I||_F < 10⁻¹⁰ after optimization (holds by
  construction: expm of Hermitian is exactly unitary up to float precision).
- **Gauge-fix:** U_1 = Γ to machine precision.
- **Plaquette at H = 0:** every U_μ = Γ; commutator trivial; W[□]_{μν} = Re(z_μν).
- **Polyakov loop:** at the center-symmetric saddle P_μ = Tr(U_μ)/N = 0.
  At H = 0 this is Tr(Γ)/N = Σ ω^k / N = 0 exactly.
- **Phase A (GW 1-matrix):** Wilson loops from the exact eigenvalue density
  agree with closed-form answers to < 10⁻⁶. Gate for proceeding to Phase B.

## Benchmark MC Data (D = 4, for Phase D comparison)

From González-Arroyo–Okawa (arXiv:1005.1981 and related), N = 289 (L = 17) with
modified flux k (k ≈ L/2):

| β = 2N/λ | ⟨plaquette⟩ at N = 289 | N → ∞ extrapolation |
|----------|------------------------|--------------------|
| 0.3560   | 0.4954(1)              | 0.495              |
| 0.3650   | 0.5482(1)              | 0.548              |
| 0.3700   | 0.5803(2)              | 0.580              |

(λ = 2N/β, so at N = 289, β = 0.356 corresponds to λ ≈ 1623.)

String tension from Creutz ratios at β = 0.356 is consistent with the infinite-
volume standard-lattice extrapolation. See arXiv:1206.0049 for details.

More precise benchmarks should be taken from:
- arXiv:1005.1981 (González-Arroyo-Okawa 2010) — modified flux, plaquette.
- arXiv:1206.0049 (González-Arroyo-Okawa 2012) — string tension, smeared Wilson
  loops.
- arXiv:1410.6405 — volume independence tests.
- arXiv:2001.10859 (García Pérez 2020) — TEK review.

## Success Criteria for Phase 3

1. Plaquette at D = 4, N = 289 matches published MC to < 1 %.
2. String tension from Creutz ratios χ(R, R) for R = 2, 3, 4 consistent with MC.
3. N → ∞ extrapolation (N ∈ {49, 121, 289, 529}) consistent with published values.
4. Runtime at N = 289 is < 1 hour (vs. days for MC).
5. The resulting matrices {Ū_μ} reproduce Wilson loops of arbitrary size by
   direct multiplication.

If successful, this provides the first explicit construction of the SU(∞) master
field for 4D lattice Yang-Mills — a problem open since Witten (1979) /
Gopakumar-Gross (1994).

## Bibliography

- Eguchi, Kawai, "Reduction of Dynamical Degrees of Freedom in the Large-N
  Gauge Theory", Phys. Rev. Lett. 48 (1982) 1063.
- Bhanot, Heller, Neuberger, "The Quenched Eguchi-Kawai Model", Phys. Lett. B
  113 (1982) 47.
- González-Arroyo, Okawa, "Twisted-Eguchi-Kawai model: A reduced model for
  large-N lattice gauge theory", Phys. Rev. D 27 (1983) 2397.
- Teper et al., "Symmetry breaking in twisted Eguchi-Kawai models", hep-th/0612097.
- González-Arroyo, Okawa, "Large N reduction with the Twisted Eguchi-Kawai
  model", arXiv:1005.1981 (JHEP 07 (2010) 043).
- González-Arroyo, Okawa, "The string tension from smeared Wilson loops at
  large N", arXiv:1206.0049.
- García Pérez, González-Arroyo, Okawa, "Perturbative contributions to Wilson
  loops in twisted lattice boxes and reduced models", arXiv:1708.00841 (2017).
- García Pérez, "The large N limit of SU(N) gauge theories", arXiv:2001.10859 (2020).
- Gopakumar, Gross, "Mastering the Master Field", hep-th/9411021.
- Witten, "The 1/N expansion in atomic and particle physics", NATO Sci. Ser. B 59 (1980).
- Kazakov, Zheng, arXiv:2203.11360 — SDP bootstrap on lattice.

## Local files

- `tek_master_field/tek.py` — core model (clock matrix, twist, links, action)
- `tek_master_field/observables.py` — plaquette, Polyakov loop, rectangular (stub)
- `tek_master_field/optimize.py` — Adam + coupling continuation
- `tek_master_field/gross_witten.py` — Phase A sanity check
- `tek_master_field/train.py` — CLI entry point
- `tek_master_field/test_tek.py` — pytest suite
