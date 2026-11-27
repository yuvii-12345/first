# The protocol, exactly as implemented

This is the spec the code follows — not a summary of the PLONK paper. Where the
implementation makes a choice the paper leaves open (or chooses differently), it
is listed in [§9](#9-deviations-from-the-plonk-paper).

---

## 1 · Parameters

| | |
|---|---|
| base field | `F_q`, `q = 21888242871839275222246405745257275088696311157297823662689037894645226208583` |
| scalar field | `F_r`, `r = 21888242871839275222246405745257275088548364400416034343698204186575808495617` |
| curve | `E/F_q : y² = x³ + 3`, `#E = r` (cofactor 1) |
| twist | `E'/F_q² : y² = x³ + 3/ξ`, `ξ = 9 + u`, `u² = −1` |
| tower | `F_q¹² = F_q⁶[w]/(w² − v)`, `F_q⁶ = F_q²[v]/(v³ − ξ)` |
| untwist | `Ψ(x', y') = (x'w², y'w³)` |
| pairing | optimal ate, loop count `6u + 2 = 29793968203157093288` |
| domain | `H = {ω⁰, …, ωⁿ⁻¹}`, `n` a power of two, `ω` a primitive `n`-th root |
| cosets | `k₁ = 7`, `k₂ = 13` |

## 2 · Arithmetization

The circuit is a table with `n` rows (padded from `numGates`). Row `i` holds
wires `(a_i, b_i, c_i)` and selectors `(qM_i, qL_i, qR_i, qO_i, qC_i)`.

```
qM·a·b + qL·a + qR·b + qO·c + qC + PI = 0            (gate)
```

Public inputs are pinned by a row whose only selector is `qL = 1`, with the
public input polynomial contributing `−v` there:

```
PI(x) = Σ_j −v_j · L_row_j(x)          ⇒   a(row_j) − v_j = 0
```

Wiring is defined by a permutation `σ` over the `3n` wire slots
(`slot·n + row`, slot ∈ {a, b, c}). Two slots that hold the same variable are
chained in a cycle; `σ₁, σ₂, σ₃` encode, for each row, the domain point of the
next occurrence:

```
x(p)   = ω^row · (1, k₁, k₂)[slot]
σ_poly(ω^row) = x(σ(p))
```

## 3 · The three identities

Interpolating each column gives `a(X), b(X), c(X), z(X), q•(X), σ•(X)`.

**Gate.**
```
gate(X) = qM·a·b + qL·a + qR·b + qO·c + qC + PI
```

**Permutation (grand product).** With `z(ω⁰) = 1`:
```
perm(X) = (a + βX + γ)(b + βk₁X + γ)(c + βk₂X + γ)·z(X)
        − (a + βσ₁ + γ)(b + βσ₂ + γ)(c + βσ₃ + γ)·z(ωX)
```

**Boundary.**
```
boundary(X) = (z(X) − 1)·L₁(X),      L₁(X) = (1/n)·Σ_{i<n} Xⁱ
```

All three vanish on `H` iff the witness satisfies the circuit. The prover packs
them with a random `α` (so a cheating prover cannot trade one violation for
another) and divides by the vanishing polynomial:

```
N(X) = gate(X) + α·perm(X) + α²·boundary(X)
t(X) = N(X) / Z_H(X),        Z_H(X) = Xⁿ − 1
```

## 4 · Degree bounds, and why the SRS is small

| polynomial | degree |
|---|---|
| `a, b, c` | `n + 1` (2 blinders each: `+r·Xᵏ·Z_H`) |
| `z` | `n + 2` (3 blinders) |
| `q•, σ•, PI` | `n − 1` |
| `perm` | `3(n+1) + (n+2) = 4n + 5` |
| `N` | `4n + 5` |
| `t` | `3n + 5` |

`t` is split at `n+2` and `2n+4` into three pieces of degree ≤ `n+1`:

```
t(X) = t₀(X) + X^{n+2}·t₁(X) + X^{2n+4}·t₂(X)
```

so the largest commitment is degree `n+1` and the SRS needs `n+2` powers
(`n+8` are generated). The quotient is computed on a coset of size
`m = 8n ≥ 4n+6`, so no aliasing occurs; `Z_H` is evaluated pointwise there.

## 5 · Messages

### Setup (universal, done once per SRS size)

`τ ←$ F_r`, then `[τ⁰]₁ … [τ^{n+1}]₁`, `[1]₂`, `[τ]₂`. τ is discarded.

`VK = (n, ω, k₁, k₂, [qM]₁, [qL]₁, [qR]₁, [qO]₁, [qC]₁, [σ₁]₁, [σ₂]₁, [σ₃]₁, [1]₂, [τ]₂, publicRows)`

### Prover

```
a, b, c            ← interpolate wires, blind, commit
β, γ               ← challenge
z                  ← grand product, blind, commit
α                  ← challenge
t₀, t₁, t₂         ← quotient, split, commit
ζ                  ← challenge
ā, b̄, c̄, σ̄₁, σ̄₂, σ̄₃, z̄, z̄ω, t̄₀, t̄₁, t̄₂, q̄M…q̄C     ← evaluations (16 scalars)
v                  ← challenge
W_ζ, W_ζω          ← batched opening proofs
```

`W_ζ` opens, at `ζ`, the batch

```
[a, b, c, σ₁, σ₂, σ₃, qM, qL, qR, qO, qC, t₀, t₁, t₂, z]
```

combined with powers of `v`, and `W_ζω = [(z − z̄ω)/(X − ζω)]₁`.

### Transcript order (must match exactly)

```
"zeroth/v1|" + domain
vk digest                     (sha-256 over the 8 VK commitments + n)
public inputs
[a]₁ [b]₁ [c]₁                → β, γ
[z]₁                          → α
[t₀]₁ [t₁]₁ [t₂]₁             → ζ
the 16 evaluations            → v
```

Every element that influences a challenge is absorbed **before** the challenge
is squeezed. Omitting one is the classic Fiat–Shamir soundness bug.

## 6 · Verification

1. **Curve check** — the nine G₁ elements lie on `y² = x³ + 3`.
2. **Identity at ζ** — the verifier computes, from public data and the 16
   claimed evaluations:

```
G   = q̄M·ā·b̄ + q̄L·ā + q̄R·b̄ + q̄O·c̄ + q̄C
P   = (ā+βζ+γ)(b̄+βk₁ζ+γ)(c̄+βk₂ζ+γ)·z̄
    − (ā+βσ̄₁+γ)(b̄+βσ̄₂+γ)(c̄+βσ̄₃+γ)·z̄ω
B   = (z̄ − 1)·L₁(ζ)
PIζ = Σ_j −v_j·L_row_j(ζ)
N   = G + PIζ + α·P + α²·B
t̄   = t̄₀ + ζ^{n+2}·t̄₁ + ζ^{2n+4}·t̄₂
check:  N == (ζⁿ − 1)·t̄
```

3. **Openings** — a single multi-pairing:

```
e(C_batch − [V]₁, [1]₂) · e(−W_ζ, [τ−ζ]₂)
  · e(C_z − [z̄ω]₁, [1]₂) · e(−W_ζω, [τζω]₂) = 1

C_batch = Σ vⁱ Cᵢ,   V = Σ vⁱ evalᵢ,   C_z = [z]₁
```

## 7 · Why forgery fails

Suppose the prover wants to accept a witness that violates some constraint, i.e.
`N − Z_H·t = Δ ≠ 0`. Its degree is at most `4n+5`. The verifier samples `ζ`
after `[t₀],[t₁],[t₂]` are committed, so `ζ` is independent of `Δ`; hence
`Δ(ζ) = 0` with probability ≤ `(4n+5)/r ≈ 2⁻²⁴⁰`.

The KZG openings then force the 16 claimed evaluations to be the true
evaluations of the committed polynomials (binding), so the verifier's scalar
computation of `N(ζ)` really is `N(ζ)`.

For the permutation specifically: the grand product `z` is committed before
`β, γ` are known, and `β, γ` are drawn after the wire commitments, so a prover
cannot choose wire values that make the product telescope without satisfying
all the copy constraints (this reduces to the standard PLONK permutation
argument).

## 8 · Zero-knowledge

`a, b, c` carry two random multiples of `Z_H` each, `z` carries three. On `H`
they are unchanged, so the constraints still hold; off `H` (in particular at
`ζ`) they are uniform. The sixteen evaluations are therefore masked, and the
commitments are hiding under the discrete-log assumption.

## 9 · Deviations from the PLONK paper

These are deliberate. Each is sound, but you should know they are there:

1. **Selectors are opened at ζ instead of being folded into a linearised
   commitment.** The proof carries `q̄M … q̄C` (5 extra scalars, 160 bytes) and the
   verifier does not need the selector polynomials — the VK is 768 bytes and the
   verifier is circuit-agnostic. The paper's linearisation avoids those five
   scalars but needs the selector commitments in the linearisation polynomial.
2. **The identity is checked as a separate scalar equality**, and the KZG
   openings are a separate pairing check. The paper folds both into one pairing
   equation. Three explicit checks are easier to reason about and to display, and
   cost one extra reusable Miller loop.
3. **`PI` enters at `α⁰` with the gate equation** (it is part of the gate
   constraint), not as its own `α`-powered term.
4. **Quotient split offsets are `n+2` and `2n+4`** rather than the paper's
   `n+2`/`2n+4`-equivalent with different blinding, because this
   implementation uses two wire blinders and three for `z`.
5. **The quotient is computed on an `8n` coset**, not `4n`, because
   `deg N = 4n+5`. Simpler and slower.

## 10 · What a production system would change

- Replace the single-party τ with a ceremony, or use a transparent scheme.
- Constant-time, Montgomery-form field arithmetic in a native/WASM backend.
- Compressed points (32 B for G₁): the proof is 1088 B here, ~600–800 B real.
- A preprocessed verifier that folds the VK into fixed pairing inputs.
- Circuit-level equivalence checks and a formal constraint audit — see
  [`SECURITY.md`](SECURITY.md).
