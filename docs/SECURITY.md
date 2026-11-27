# Security notes

**This is a teaching implementation. Do not use it to protect anything.**

The protocol is real — the field, curve, pairing, NTT, KZG and PLONK code all do
what they claim, and the tests exercise forgery attempts. But "the protocol is
correct" and "this software is safe" are different sentences, and the gap is
deliberate.

## Threat model

What the implementation *does* provide, in the honest-execution model:

| Property | Status |
|---|---|
| **Completeness** — a true statement with a valid witness always verifies | yes, tested |
| **Soundness** — a false statement cannot produce an accepted proof (except with negligible probability) | yes, `O(n)/r` error, tested by tampering |
| **Zero-knowledge** — the proof reveals nothing but the truth of the statement | yes, via `Z_H` blinding of `a,b,c,z`, assuming the discrete-log assumption holds |
| **Binding** — a commitment cannot be opened to two values | yes, unless τ is known |
| **Proof of knowledge** | by the standard KZG/PLONK argument |

What it does **not** provide: side-channel resistance, key management, a
trusted-setup ceremony, resource limits, or any defence against a malicious
*verifier* beyond what the protocol itself gives.

## Deliberate simplifications

1. **No trusted-setup ceremony.** τ is generated with
   `crypto.getRandomValues`, used to build the SRS, and dropped. One party knows
   τ for as long as the SRS takes to build; a real deployment needs a
   multi-party computation, or a transparent commitment scheme.
2. **Not constant-time.** `BigInt` arithmetic branches on values, `%` and `/`
   have data-dependent timing, and the field inversion uses a variable-time
   extended Euclidean algorithm. Anyone who can measure the prover's timing may
   learn something about the witness. Do not prove secrets on a shared machine.
3. **Uncompressed points.** 64 bytes per G₁ element. Real systems compress to 32.
4. **Naive final exponentiation.** A plain square-and-multiply by
   `(q¹²−1)/r`, not the standard easy-part/hard-part split. Correct, ~3× slower.
5. **Affine line functions in the Miller loop**, lifted into `F_q¹²`. Chosen for
   auditability over speed.
6. **No subgroup check on G₂ inputs** — the only G₂ points that enter
   verification come from the SRS, which is trusted by construction. If you ever
   accept G₂ points from a prover, add the check (`isInSubgroupG2`).
7. **The verifier trusts its own VK.** There is no binding between the VK and a
   circuit name in the transcripts beyond a SHA-256 digest of the commitments.

## What is actually checked at verification

1. All nine G₁ elements are on the curve.
2. The full PLONK identity holds at a Fiat–Shamir point `ζ`
   (gate + public inputs + permutation + boundary).
3. All sixteen claimed evaluations are consistent with the commitments, via a
   batch KZG opening at `ζ` and a second opening at `ζω`.

Failing any one rejects the proof. The Soundness tab mutates each in turn so you
can see which check fires.

## Failure modes this codebase is built to expose

- **A missing constraint.** `qM·a·b − c = 0` deleted ⇒ `3 × 11 = 34` verifies.
  Try it: Soundness → *delete the x·y = z constraint*.
- **A wrong witness.** The prover audits every gate equation before proving and
  refuses, with the offending row numbers.
- **A modified evaluation.** Any of the sixteen scalars is bound by KZG; +1 to
  any of them fails the opening check.
- **A modified commitment.** Swapping two commitments fails the opening check.
- **A modified statement.** Public inputs are absorbed into the transcript and
  into `PI`; changing one changes `ζ` and the identity check fails.

Not exposed, but worth knowing:

- **Forgetting to absorb a commitment** before squeezing a challenge lets the
  prover grind the challenge. There is a comment at every `absorb` call for this
  reason.
- **Reusing `ζ` across circuits** without a domain separator enables
  batch-opening forgeries; the transcript is domain-separated by a VK digest.
- **Leaking the witness through the proof size** — all circuits here are padded
  to a power of two, which leaks the *number of gates*, not the values.

## Reporting

This is a demonstration project with no users to protect. If you find a bug that
would make the pedagogy wrong — a formula that does not check out, a tamper case
that slips through — that is a real bug and worth an issue.
