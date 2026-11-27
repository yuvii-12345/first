import { useEffect, useRef, useState, type ReactNode } from "react";

export function Panel({
  title,
  right,
  children,
  tight,
  className,
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  tight?: boolean;
  className?: string;
}) {
  return (
    <section className={`panel ${className ?? ""}`}>
      {title !== undefined && (
        <header className="panel-head">
          <h3>{title}</h3>
          <div className="spacer" />
          {right}
        </header>
      )}
      <div className={tight ? "panel-body tight" : "panel-body"}>{children}</div>
    </section>
  );
}

export function Badge({ children, tone }: { children: ReactNode; tone?: "ok" | "bad" | "warn" }) {
  return <span className={`badge ${tone ?? ""}`}>{children}</span>;
}

export function Stat({ k, v, sub }: { k: string; v: ReactNode; sub?: string }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className="v">
        {v}
        {sub && <small>{sub}</small>}
      </div>
    </div>
  );
}

export function Stats({ children }: { children: ReactNode }) {
  return <div className="stats">{children}</div>;
}

export function CheckRow({ check }: { check: { label: string; ok: boolean; detail: string } }) {
  return (
    <div className="check" data-ok={check.ok}>
      <span className="icon">{check.ok ? "✓" : "✕"}</span>
      <div>
        <div className="label">{check.label}</div>
        <div className="detail">{check.detail}</div>
      </div>
      <Badge tone={check.ok ? "ok" : "bad"}>{check.ok ? "pass" : "fail"}</Badge>
    </div>
  );
}

export function Hex({ value, className }: { value: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      timer.current = window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <div style={{ position: "relative" }}>
      <div className={`hex ${className ?? ""}`}>{value}</div>
      <button
        className="btn small ghost"
        onClick={copy}
        style={{ position: "absolute", top: 6, right: 6, background: "var(--panel)" }}
        type="button"
      >
        {copied ? "copied" : "copy"}
      </button>
    </div>
  );
}

export function KV({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <dl className="kv">
      {items.map(([k, v], i) => (
        <div key={i} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <div className="lab">
        <span>{label}</span>
        {hint && <span style={{ textTransform: "none", letterSpacing: 0 }}>{hint}</span>}
      </div>
      {children}
    </label>
  );
}

export function Toggle({
  items,
  value,
  onChange,
  disabled,
}: {
  items: Array<{ id: string; label: string }>;
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  return (
    <div style={{ display: "inline-flex", border: "1px solid var(--line-2)", borderRadius: 3, overflow: "hidden" }}>
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          disabled={disabled}
          onClick={() => onChange(it.id)}
          className="btn small"
          style={{
            border: 0,
            borderRadius: 0,
            borderRight: "1px solid var(--line-2)",
            background: value === it.id ? "var(--accent-soft)" : "transparent",
            color: value === it.id ? "var(--accent)" : "var(--muted)",
            padding: "6px 11px",
          }}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <label className="switch" style={disabled ? { opacity: 0.5, pointerEvents: "none" } : undefined}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="track" />
      <span>{label}</span>
    </label>
  );
}

export function SectionHead({
  kicker,
  title,
  lead,
  right,
}: {
  kicker: string;
  title: ReactNode;
  lead?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="section-head">
      <div>
        <div className="kicker">{kicker}</div>
        <h1>{title}</h1>
        {lead && (
          <p className="lead" style={{ marginTop: 8, marginBottom: 0 }}>
            {lead}
          </p>
        )}
      </div>
      {right && <div style={{ flex: "none" }}>{right}</div>}
    </div>
  );
}

export function Spinner() {
  return <span className="spinner" />;
}

export function fmtMs(ms: number): string {
  if (!isFinite(ms)) return "—";
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  return `${(b / 1024).toFixed(1)} kB`;
}

export function shortHex(v: string, head = 10, tail = 6): string {
  if (!v.startsWith("0x")) return v;
  const body = v.slice(2);
  if (body.length <= head + tail + 2) return v;
  return `0x${body.slice(0, head)}…${body.slice(-tail)}`;
}

export function shortDec(v: string, chars = 12): string {
  return v.length <= chars ? v : `${v.slice(0, chars)}…`;
}
