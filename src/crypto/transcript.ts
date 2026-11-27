/**
 * Fiat–Shamir transcript.
 *
 * An interactive SNARK needs the verifier to send random challenges. We make
 * it non-interactive by hashing everything said so far: the "random oracle"
 * replaces the verifier. Every challenge is domain-separated by a label, so
 * that two different challenges can never collide even if they are squeezed at
 * the same point in the protocol.
 *
 * Rule of thumb: **every** group element and every evaluation that influences
 * a challenge must be absorbed before the challenge is squeezed. Forget one and
 * the prover can grind it — this is the classic Fiat–Shamir soundness bug.
 */
import { R } from "./field";
import { concatBytes, sha256, utf8 } from "./sha256";

function bytesOfBigInt(v: bigint, len = 32): Uint8Array {
  const out = new Uint8Array(len);
  let x = v;
  for (let i = len - 1; i >= 0; i--) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return out;
}

export class Transcript {
  private state: Uint8Array;
  private counter = 0;

  constructor(domainSep: string) {
    this.state = sha256(concatBytes(utf8("zeroth/v1|"), utf8(domainSep)));
  }

  private update(tag: string, payload: Uint8Array): void {
    const len = bytesOfBigInt(BigInt(payload.length), 8);
    this.state = sha256(concatBytes(this.state, utf8("|" + tag + "|"), len, payload));
  }

  absorbBytes(tag: string, data: Uint8Array): this {
    this.update(tag, data);
    return this;
  }

  absorbField(tag: string, v: bigint): this {
    return this.absorbBytes(tag, bytesOfBigInt(v));
  }

  absorbFields(tag: string, vs: bigint[]): this {
    const buf = new Uint8Array(vs.length * 32);
    vs.forEach((v, i) => buf.set(bytesOfBigInt(v), i * 32));
    return this.absorbBytes(tag, buf);
  }

  absorbPoint(tag: string, p: Uint8Array): this {
    return this.absorbBytes(tag, p);
  }

  absorbString(tag: string, s: string): this {
    return this.absorbBytes(tag, utf8(s));
  }

  /** Squeeze a uniformly random scalar field element. */
  challenge(label: string): bigint {
    for (let attempt = 0; attempt < 1000; attempt++) {
      const h = sha256(
        concatBytes(
          this.state,
          utf8("|challenge|" + label + "|" + this.counter + "|" + attempt + "|"),
        ),
      );
      let v = 0n;
      for (const b of h) v = (v << 8n) | BigInt(b);
      // Reject to remove modulo bias (h is 256 bits, r is 254 bits).
      if (v < R) {
        this.counter++;
        this.state = sha256(concatBytes(this.state, h));
        return v;
      }
    }
    throw new Error("challenge: rejection sampling failed");
  }

  /** Squeeze several independent challenges at once. */
  challenges(label: string, count: number): bigint[] {
    return Array.from({ length: count }, (_, i) => this.challenge(`${label}/${i}`));
  }

  /** A snapshot of the running digest, for the inspector UI. */
  digest(): string {
    return Array.from(this.state, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  clone(): Transcript {
    const t = new Transcript("clone");
    t.state = this.state.slice();
    t.counter = this.counter;
    return t;
  }
}

export { bytesOfBigInt };
