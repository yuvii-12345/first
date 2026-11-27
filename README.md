# zeroth

**A zk-SNARK built from bare arithmetic.** PLONK over BN254, with KZG polynomial
commitments, a Fiat–Shamir transcript, a mini-DSL, and an inspector that shows
every intermediate value of a real proof.

**Live: [zeroth-opal.vercel.app](https://zeroth-opal.vercel.app)**

```bash
npm install
npm run dev      # http://localhost:5173
npm run demo     # the same thing in a terminal
npm test         # 67 tests, including the soundness ones
```

No cryptography dependencies. `react` and `vite` are the only things in
`node_modules` that matter. Field arithmetic, the curve, the pairing, the NTT,
KZG and the protocol are all in `src/` and all readable.

---

## Why this exists

> *"A single missing constraint turns a proof system from cryptographically
> sound into completely forgeable."*

That sentence is the whole project. The cryptography is the easy part once you
have the arithmetic; the hard part is that **soundness is global**. Every one of
the thousands of intended constraints has to be present, wired correctly, and
checked at the right point in the protocol — otherwise the prover can certify a
false statement, and the verifier will look at it and say *valid*.

So this repository does two things at once:

1. It implements the stack honestly, from `%` upwards, with tests.
2. It makes the failure modes **executable**. In the Soundness tab you can
   delete a single constraint and watch a 1088-byte proof of `3 × 11 = 34`
   sail through verification. Nothing crashes. Every check passes. That is the
   point.

---

## The ladder

Each layer exists because the layer below it cannot answer the question above.

| Layer | File | What it buys you |
|---|---|---|
| Prime field `F_q`, `F_r` | `src/crypto/field.ts` | arithmetic with inverses, 254 bits of BigInt |
| `F_q² ⊂ F_q⁶ ⊂ F_q¹²` | `src/crypto/field.ts` | the tower a pairing needs |
| Curve G₁ / G₂, Pippenger | `src/crypto/curve.ts` | a group where discrete log is hard |
| Optimal ate pairing | `src/crypto/pairing.ts` | checking equations *in the exponent* |
| NTT, cosets, Lagrange | `src/crypto/ntt.ts`, `src/crypto/poly.ts` | polynomial math in O(n log n) |
| KZG commitments | `src/crypto/kzg.ts` | a short fingerprint of a polynomial |
| Arithmetization | `src/plonk/circuit.ts` | gates + a permutation for the wiring |
| Prover / verifier | `src/plonk/plonk.ts` | the protocol itself |
| DSL | `src/dsl/dsl.ts` | `secret x = 7 … assert x*y == z` |
| UI | `src/app/` | the inspector, the labs |

The proof is **9 G₁ points + 16 scalars = 1088 bytes**. The verifier key is
768 bytes. Verification is an O(1) scalar identity plus four pairings.

---

## The five screens

- **Prove** — pick a statement, edit the private witness, run it. The pipeline
  lights up stage by stage with real timings; the verdict shows the three
  verifier checks (curve membership, the identity at ζ, the KZG openings).
- **Inspector** — every commitment, challenge and evaluation of the run you just
  did. This is the page to read if you want to know what a proof *is*.
- **Circuit** — the gate table, the variable table, and the copy-constraint
  cycles. This is where you see that the wiring is a separate argument from the
  gates.
- **Soundness** — three labs: delete a constraint (forgeability), tamper with a
  proof element (which check catches it), and read what each check actually
  buys.
- **Primer** — the math, rung by rung, with live checks you can run (sample
  inverses, verify bilinearity of the pairing in front of you).

## The statements

| Scenario | Statement | Rows |
|---|---|---|
| Multiplication | `x · y = z`, `x, y` secret | 8 |
| MiMC preimage | `MiMC(x, k) = y`, `x` secret | 512 |
| Sudoku 4×4 | "I know a completed grid agreeing with the givens" | 512 |
| Sudoku 9×9 | the same, 729 one-hot wires and 3,677 constraints | 4,096 |
| DSL | whatever you write | — |

MiMC is used instead of SHA-256 on purpose: bit-oriented hashes cost tens of
thousands of gates in a SNARK, while a field-native hash costs four gates per
round. Sudoku is one-hot encoded, and the *per-cell* "exactly one digit"
constraint is the one the Soundness lab deletes.

## Numbers, measured in this browser

| | 4×4 Sudoku | MiMC (110 rounds) | 9×9 Sudoku |
|---|---|---|---|
| domain | 512 | 1024 | 4096 |
| preprocessing | 1.2 s | 2.5 s | 6–10 s |
| proving | 0.8 s | 2.2 s | 7 s |
| verifying | 0.3 s | 0.3 s | 0.3 s |

These are BigInt timings — roughly two orders of magnitude slower than a
Montgomery-form native implementation. They are honest, not impressive.

---

## What this is not

- **No ceremony.** τ is sampled in the browser with `crypto.getRandomValues` and
  thrown away. A real deployment needs a multi-party powers-of-tau ceremony (or
  a transparent setup).
- **Not constant-time.** `BigInt` is not a side-channel-safe primitive.
- **Not audited.** This is a teaching implementation of a well-understood
  protocol, written to be read. See [`docs/SECURITY.md`](docs/SECURITY.md) for
  the full list of deliberate simplifications.

Start with [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the equations, or just open
the app and break something. [`docs/DEPLOY.md`](docs/DEPLOY.md) covers the Vercel
setup.
