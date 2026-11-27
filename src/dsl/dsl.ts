/**
 * A tiny DSL for PLONK circuits.
 *
 *   secret x = 7
 *   secret y = 13
 *   public product = 91
 *   assert x * y == product
 *
 * `secret` values are the prover's private witness; `public` values are the
 * statement (visible to the verifier); `let` introduces an intermediate wire;
 * `assert` emits constraints. Every arithmetic operation you write becomes a
 * row of the gate table, and the compiler's variable bookkeeping is what makes
 * the copy constraints (the permutation argument) meaningful.
 */
import { FrField } from "../crypto/field";
import { CircuitBuilder, type CompiledCircuit, type VarId } from "../plonk/circuit";

// -- lexer ------------------------------------------------------------------

type TokKind = "num" | "ident" | "op" | "kw" | "eof";

interface Token {
  kind: TokKind;
  text: string;
  line: number;
  col: number;
}

const KEYWORDS = new Set(["secret", "public", "let", "assert"]);

export class DslError extends Error {
  constructor(
    message: string,
    readonly line: number,
    readonly col: number,
  ) {
    super(`line ${line}:${col}: ${message}`);
    this.name = "DslError";
  }
}

function lex(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const advance = (n = 1) => {
    for (let k = 0; k < n; k++) {
      if (src[i] === "\n") {
        line++;
        col = 1;
      } else col++;
      i++;
    }
  };
  while (i < src.length) {
    const ch = src[i];
    if (ch === " " || ch === "\t" || ch === "\r") {
      advance();
      continue;
    }
    if (ch === "\n") {
      advance();
      continue;
    }
    if (ch === "/" && src[i + 1] === "/") {
      while (i < src.length && src[i] !== "\n") advance();
      continue;
    }
    if (/[0-9]/.test(ch)) {
      const start = i;
      while (i < src.length && /[0-9_]/.test(src[i])) advance();
      tokens.push({ kind: "num", text: src.slice(start, i).replace(/_/g, ""), line, col: col - (i - start) });
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      const start = i;
      const c0 = col;
      while (i < src.length && /[A-Za-z0-9_]/.test(src[i])) advance();
      const text = src.slice(start, i);
      tokens.push({ kind: KEYWORDS.has(text) ? "kw" : "ident", text, line, col: c0 });
      continue;
    }
    if (ch === "=" && src[i + 1] === "=") {
      tokens.push({ kind: "op", text: "==", line, col });
      advance(2);
      continue;
    }
    if (ch === "!" && src[i + 1] === "=") {
      tokens.push({ kind: "op", text: "!=", line, col });
      advance(2);
      continue;
    }
    if ("+-*()=;".includes(ch)) {
      tokens.push({ kind: "op", text: ch, line, col });
      advance();
      continue;
    }
    throw new DslError(`unexpected character '${ch}'`, line, col);
  }
  tokens.push({ kind: "eof", text: "<eof>", line, col });
  return tokens;
}

// -- parser -----------------------------------------------------------------

type Expr =
  | { t: "num"; v: bigint; line: number }
  | { t: "ident"; v: string; line: number }
  | { t: "bin"; op: "+" | "-" | "*"; l: Expr; r: Expr; line: number }
  | { t: "neg"; e: Expr; line: number };

type Stmt =
  | { t: "declare"; kind: "secret" | "public" | "let"; name: string; expr: Expr; line: number }
  | { t: "assertEq"; l: Expr; r: Expr; line: number }
  | { t: "assertNonZero"; e: Expr; line: number };

class Parser {
  private pos = 0;
  constructor(private readonly toks: Token[]) {}

  private peek(): Token {
    return this.toks[this.pos];
  }
  private next(): Token {
    return this.toks[this.pos++];
  }
  private expect(text: string): Token {
    const t = this.peek();
    if (t.text !== text) throw new DslError(`expected '${text}' but found '${t.text}'`, t.line, t.col);
    return this.next();
  }
  private skipSeparators(): void {
    while (this.peek().text === ";" || this.peek().text === "\n") this.next();
  }

  parseProgram(): Stmt[] {
    const stmts: Stmt[] = [];
    this.skipSeparators();
    while (this.peek().kind !== "eof") {
      stmts.push(this.parseStmt());
      this.skipSeparators();
    }
    return stmts;
  }

  private parseStmt(): Stmt {
    const t = this.peek();
    if (t.kind === "kw" && (t.text === "secret" || t.text === "public" || t.text === "let")) {
      this.next();
      const name = this.expectIdent();
      this.expect("=");
      const expr = this.parseExpr();
      return { t: "declare", kind: t.text as "secret" | "public" | "let", name: name.text, expr, line: t.line };
    }
    if (t.kind === "kw" && t.text === "assert") {
      this.next();
      const left = this.parseExpr();
      const op = this.peek();
      if (op.text === "==") {
        this.next();
        const right = this.parseExpr();
        return { t: "assertEq", l: left, r: right, line: t.line };
      }
      if (op.text === "!=") {
        this.next();
        const zero = this.parseExpr();
        return { t: "assertNonZero", e: left, line: t.line } as Stmt;
      }
      throw new DslError(`expected '==' or '!=' in assert, found '${op.text}'`, op.line, op.col);
    }
    throw new DslError(`expected a statement (secret / public / let / assert), found '${t.text}'`, t.line, t.col);
  }

  private expectIdent(): Token {
    const t = this.peek();
    if (t.kind !== "ident") throw new DslError(`expected an identifier, found '${t.text}'`, t.line, t.col);
    return this.next();
  }

  parseExpr(): Expr {
    let left = this.parseTerm();
    for (;;) {
      const t = this.peek();
      if (t.text === "+" || t.text === "-") {
        this.next();
        const right = this.parseTerm();
        left = { t: "bin", op: t.text as "+" | "-", l: left, r: right, line: t.line };
      } else break;
    }
    return left;
  }

  private parseTerm(): Expr {
    let left = this.parseUnary();
    for (;;) {
      const t = this.peek();
      if (t.text === "*") {
        this.next();
        const right = this.parseUnary();
        left = { t: "bin", op: "*", l: left, r: right, line: t.line };
      } else break;
    }
    return left;
  }

  private parseUnary(): Expr {
    const t = this.peek();
    if (t.text === "-") {
      this.next();
      return { t: "neg", e: this.parseUnary(), line: t.line };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): Expr {
    const t = this.next();
    if (t.kind === "num") return { t: "num", v: BigInt(t.text), line: t.line };
    if (t.kind === "ident") return { t: "ident", v: t.text, line: t.line };
    if (t.text === "(") {
      const e = this.parseExpr();
      this.expect(")");
      return e;
    }
    throw new DslError(`unexpected '${t.text}' in expression`, t.line, t.col);
  }
}

// -- compiler ---------------------------------------------------------------

export interface DslResult {
  circuit: CompiledCircuit;
  /** Public inputs, in declaration order. */
  publicValues: bigint[];
  publicNames: string[];
  statements: number;
}

export function compileDsl(source: string): DslResult {
  const stmts = new Parser(lex(source)).parseProgram();
  const b = new CircuitBuilder();
  const vars = new Map<string, VarId>();
  const publicValues: bigint[] = [];
  const publicNames: string[] = [];

  const constValue = (e: Expr): bigint => {
    switch (e.t) {
      case "num":
        return FrField.reduce(e.v);
      case "neg":
        return FrField.neg(constValue(e.e));
      case "bin": {
        const l = constValue(e.l);
        const r = constValue(e.r);
        if (e.op === "+") return FrField.add(l, r);
        if (e.op === "-") return FrField.sub(l, r);
        return FrField.mul(l, r);
      }
      case "ident": {
        const id = vars.get(e.v);
        if (id === undefined) throw new DslError(`unknown variable '${e.v}'`, e.line, 1);
        return FrField.reduce(b.variables[id].value);
      }
    }
  };

  const emit = (e: Expr): VarId => {
    switch (e.t) {
      case "num":
        return b.constant(`k${FrField.reduce(e.v).toString(16).slice(0, 6)}`, e.v);
      case "ident": {
        const id = vars.get(e.v);
        if (id === undefined) throw new DslError(`unknown variable '${e.v}'`, e.line, 1);
        return id;
      }
      case "neg": {
        const inner = emit(e.e);
        return b.sub(b.zero, inner, "neg");
      }
      case "bin": {
        const l = emit(e.l);
        const r = emit(e.r);
        if (e.op === "*") return b.mul(l, r, "mul");
        if (e.op === "+") return b.add(l, r, "add");
        return b.sub(l, r, "sub");
      }
    }
  };

  for (const s of stmts) {
    if (s.t === "declare") {
      const value = constValue(s.expr);
      if (s.kind === "secret") {
        vars.set(s.name, b.secret(s.name, value));
      } else if (s.kind === "public") {
        vars.set(s.name, b.publicInput(s.name, value));
        publicValues.push(FrField.reduce(value));
        publicNames.push(s.name);
      } else {
        // `let` may reference any variable and emits gates
        const before = new Set(vars.keys());
        void before;
        const id = emit(s.expr);
        vars.set(s.name, id);
        b.variables[id].name = s.name;
      }
    } else if (s.t === "assertEq") {
      const l = emit(s.l);
      const r = emit(s.r);
      b.assertEqual(l, r, "assert ==");
    } else {
      const e = emit(s.e);
      b.assertNonZero(e);
    }
  }

  return { circuit: b.finalize(), publicValues, publicNames, statements: stmts.length };
}

export const EXAMPLE_PROGRAMS: Array<{ name: string; source: string; note: string }> = [
  {
    name: "multiplication",
    note: "The hello-world of zero-knowledge: prove you know the factors of a public product.",
    source: `// The prover knows two numbers. The verifier only sees their product.
secret x = 7
secret y = 13
public product = 91

assert x * y == product`,
  },
  {
    name: "square root",
    note: "Prove you know a square root of a public value, without revealing which root.",
    source: `secret root = 123456789
public square = 15241578750190521

assert root * root == square`,
  },
  {
    name: "boolean select",
    note: "A one-bit secret decides which of two public values is revealed — and the proof hides which.",
    source: `secret flag = 1
secret a = 111
secret b = 222

assert flag * (flag - 1) == 0
let out = flag * (a - b) + b

public chosen = 111
assert out == chosen`,
  },
  {
    name: "polynomial",
    note: "Evaluate x^3 + 2x + 5 in the exponent — arithmetic circuits are polynomial machines.",
    source: `secret x = 9
let x2 = x * x
let x3 = x2 * x
let fx = x3 + 2 * x + 5

public y = 752
assert fx == y`,
  },
];
