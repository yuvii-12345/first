// LUMEN Canvas Renderer — interprets 14-dim physics params into visuals
// Procedural SDF + shading, runs at 60fps on CPU
// Each archetype has distinct visual program, blended by archetype parameter

export type FrameParams = Float32Array; // 14 dims

const ARCHETYPES = ["bounce", "rotate", "expand", "wave", "orbit", "particles", "gradient", "orbit2"] as const;

export interface RenderOptions {
  width: number;
  height: number;
  fps?: number;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function lerpColor(c1: number[], c2: number[], t: number): number[] {
  return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
}
function clamp(v: number, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, v));
}

// Draw single frame to canvas
export function renderFrame(
  ctx: CanvasRenderingContext2D,
  params: FrameParams,
  tNorm: number, // 0..1 overall progress
  width: number,
  height: number,
  styleHint?: string
) {
  const xPos = params[0];
  const yPos = params[1];
  const scale = params[2];
  const rotation = params[3];
  const phase = params[4];
  const bgMix = params[5];
  const col1 = [params[6], params[7], params[8]];
  const col2 = [params[9], params[10], params[11]];
  const glow = params[12];
  const archetypeBlend = params[13];

  const archIdx = Math.round(clamp(archetypeBlend, 0, 1) * (ARCHETYPES.length - 1));
  const archetype = ARCHETYPES[archIdx] as string;

  // Map to archetype for consistent visuals even if blend is intermediate
  // We use hard archetype selection for clarity, but we also use blend to lerp visuals subtly
  ctx.clearRect(0, 0, width, height);

  // Background base
  const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
  const bgA = lerpColor([5, 5, 8], [col2[0] * 30, col2[1] * 30, col2[2] * 45], 0.15 + bgMix * 0.2);
  const bgB = lerpColor([10, 10, 16], [col1[0] * 20, col1[1] * 20, col1[2] * 35], 0.12);
  bgGrad.addColorStop(0, `rgb(${bgA[0] | 0},${bgA[1] | 0},${bgA[2] | 0})`);
  bgGrad.addColorStop(1, `rgb(${bgB[0] | 0},${bgB[1] | 0},${bgB[2] | 0})`);
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, width, height);

  // Vignette
  const vign = ctx.createRadialGradient(width / 2, height / 2, 0, width / 2, height / 2, Math.max(width, height) * 0.9);
  vign.addColorStop(0, "rgba(0,0,0,0)");
  vign.addColorStop(1, "rgba(0,0,0,0.45)");
  ctx.fillStyle = vign;
  ctx.fillRect(0, 0, width, height);

  // Archetype rendering
  const cx = width / 2 + xPos * width * 0.38;
  const cy = height / 2 + yPos * height * 0.38;
  const s = Math.min(width, height);

  ctx.save();

  if (archetype === "bounce") {
    // Ball with shadow & highlight
    const ballR = scale * s * 0.42;
    const animPhase = tNorm * Math.PI * 4;
    const bounceY = Math.abs(Math.sin(animPhase + phase)) * -height * 0.22;
    const bx = cx + Math.sin(tNorm * 1.5 + phase) * width * 0.08;
    const by = cy + bounceY;

    // Shadow
    ctx.globalAlpha = 0.25 * (1 - Math.abs(bounceY) / (height * 0.25));
    ctx.fillStyle = "rgba(0,0,0,1)";
    ctx.beginPath();
    ctx.ellipse(cx, cy + height * 0.32, ballR * 0.9, ballR * 0.35, 0, 0, Math.PI * 2);
    ctx.fill();

    // Ball gradient
    ctx.globalAlpha = 1;
    const grad = ctx.createRadialGradient(bx - ballR * 0.3, by - ballR * 0.3, ballR * 0.2, bx, by, ballR);
    grad.addColorStop(0, `rgb(${clamp(col1[0] * 255 + 60, 0, 255) | 0},${clamp(col1[1] * 255 + 60, 0, 255) | 0},${clamp(col1[2] * 255 + 60, 0, 255) | 0})`);
    grad.addColorStop(0.6, `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`);
    grad.addColorStop(1, `rgb(${(col1[0] * 120) | 0},${(col1[1] * 120) | 0},${(col1[2] * 120) | 0})`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(bx, by, ballR, 0, Math.PI * 2);
    ctx.fill();

    // Highlight
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.beginPath();
    ctx.arc(bx - ballR * 0.3, by - ballR * 0.28, ballR * 0.22, 0, Math.PI * 2);
    ctx.fill();

    // Motion trail
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`;
    ctx.beginPath();
    ctx.arc(bx, by + ballR * 0.15, ballR * 0.65, 0, Math.PI * 2);
    ctx.fill();
  } else if (archetype === "rotate") {
    const sqSize = scale * s * 0.85;
    ctx.translate(cx, cy);
    ctx.rotate(rotation);
    // Back face
    ctx.fillStyle = `rgba(${(col2[0] * 255) | 0},${(col2[1] * 255) | 0},${(col2[2] * 255) | 0},0.9)`;
    ctx.fillRect(-sqSize / 2 - 6, -sqSize / 2 - 6, sqSize, sqSize);
    // Front
    const grad = ctx.createLinearGradient(-sqSize / 2, -sqSize / 2, sqSize / 2, sqSize / 2);
    grad.addColorStop(0, `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`);
    grad.addColorStop(1, `rgb(${(col1[0] * 160) | 0},${(col1[1] * 160) | 0},${(col1[2] * 160) | 0})`);
    ctx.fillStyle = grad;
    ctx.fillRect(-sqSize / 2, -sqSize / 2, sqSize, sqSize);
    // Inner cut
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = "white";
    for (let i = 0; i < 3; i++) {
      ctx.fillRect(-sqSize / 2 + i * 8, -sqSize / 2, 2, sqSize);
      ctx.fillRect(-sqSize / 2, -sqSize / 2 + i * 8, sqSize, 2);
    }
    ctx.globalAlpha = 1;
    // Glow
    ctx.shadowColor = `rgba(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0},0.6)`;
    ctx.shadowBlur = 18 * glow;
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.strokeRect(-sqSize / 2, -sqSize / 2, sqSize, sqSize);
    ctx.shadowBlur = 0;
  } else if (archetype === "expand") {
    const r = scale * s * 0.72;
    const pulse = 0.5 + 0.5 * Math.sin(tNorm * Math.PI * 4 + phase);
    const curR = r * (0.6 + pulse * 0.4);
    // Glow
    ctx.globalAlpha = 0.18 * glow + 0.08;
    ctx.fillStyle = `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`;
    ctx.beginPath();
    ctx.arc(cx, cy, curR * 1.35, 0, Math.PI * 2);
    ctx.fill();
    // Outer ring
    ctx.globalAlpha = 0.95;
    ctx.strokeStyle = `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`;
    ctx.lineWidth = Math.max(1.5, s * 0.015);
    ctx.beginPath();
    ctx.arc(cx, cy, curR, 0, Math.PI * 2);
    ctx.stroke();
    // Inner fill
    const innerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, curR);
    innerGrad.addColorStop(0, `rgba(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0},0.35)`);
    innerGrad.addColorStop(0.7, `rgba(${(col2[0] * 255) | 0},${(col2[1] * 255) | 0},${(col2[2] * 255) | 0},0.2)`);
    innerGrad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = innerGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, curR, 0, Math.PI * 2);
    ctx.fill();
    // Secondary echo
    ctx.globalAlpha = 0.22 * (1 - pulse * 0.5);
    ctx.strokeStyle = `rgb(${(col2[0] * 255) | 0},${(col2[1] * 255) | 0},${(col2[2] * 255) | 0})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, curR * (1.18 + pulse * 0.12), 0, Math.PI * 2);
    ctx.stroke();
  } else if (archetype === "wave") {
    // Wave displacement field
    ctx.translate(0, 0);
    // Clip to canvas
    for (let y = 0; y < height; y += 2) {
      const wave = Math.sin((0 / width) * 6 + tNorm * 9 + phase) * 8; // will be dynamic per x below
      // Actually draw scanline with wave
      // We'll draw horizontal bands with sine
    }
    // More efficient: draw via path segments
    const bands = 7;
    for (let b = 0; b < bands; b++) {
      const yBase = (height / (bands + 1)) * (b + 1);
      ctx.beginPath();
      ctx.moveTo(0, yBase);
      for (let x = 0; x <= width; x += 4) {
        const wx = Math.sin((x / width) * 10 + tNorm * 6 + b * 0.8 + phase) * 18;
        const wy = yBase + wx + Math.sin(tNorm * 3 + x * 0.01) * 3;
        ctx.lineTo(x, wy);
      }
      const alpha = 0.18 + b * 0.04 + glow * 0.12;
      ctx.strokeStyle = `rgba(${(lerp(col1[0], col2[0], b / bands) * 255) | 0},${(lerp(col1[1], col2[1], b / bands) * 255) | 0},${(lerp(col1[2], col2[2], b / bands) * 255) | 0},${alpha})`;
      ctx.lineWidth = 2 + b * 0.6;
      ctx.stroke();
      // fill under
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      ctx.fillStyle = `rgba(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0},${0.04 + b * 0.008})`;
      ctx.fill();
    }
  } else if (archetype === "orbit") {
    const orbitR = s * 0.32;
    const a1 = tNorm * Math.PI * 2 * 1.3 + phase;
    const a2 = a1 + Math.PI;
    // Trails
    ctx.globalAlpha = 0.22;
    ctx.strokeStyle = `rgba(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0},0.35)`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, orbitR, orbitR * 0.62, 0, 0, Math.PI * 2);
    ctx.stroke();
    // Planets
    const trailCount = 6;
    for (let k = trailCount; k >= 0; k--) {
      const tt = Math.max(0, tNorm - k * 0.03);
      const aa = tt * Math.PI * 2 * 1.3 + phase;
      const px = cx + Math.cos(aa) * orbitR;
      const py = cy + Math.sin(aa) * orbitR * 0.62;
      ctx.globalAlpha = (1 - k / trailCount) * 0.12;
      ctx.fillStyle = `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`;
      ctx.beginPath();
      ctx.arc(px, py, 4 + (1 - k / trailCount) * 2, 0, Math.PI * 2);
      ctx.fill();
    }
    const p1x = cx + Math.cos(a1) * orbitR;
    const p1y = cy + Math.sin(a1) * orbitR * 0.62;
    const p2x = cx + Math.cos(a2) * orbitR;
    const p2y = cy + Math.sin(a2) * orbitR * 0.62;
    // Body 1
    ctx.globalAlpha = 1;
    const g1 = ctx.createRadialGradient(p1x - 3, p1y - 3, 2, p1x, p1y, scale * s * 0.12);
    g1.addColorStop(0, `rgb(${clamp(col1[0] * 255 + 50, 0, 255) | 0},${clamp(col1[1] * 255 + 50, 0, 255) | 0},${clamp(col1[2] * 255 + 50, 0, 255) | 0})`);
    g1.addColorStop(1, `rgb(${(col1[0] * 180) | 0},${(col1[1] * 180) | 0},${(col1[2] * 180) | 0})`);
    ctx.fillStyle = g1;
    ctx.shadowColor = `rgba(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0},0.6)`;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(p1x, p1y, scale * s * 0.12, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    // Body 2
    const g2 = ctx.createRadialGradient(p2x - 2, p2y - 2, 1, p2x, p2y, scale * s * 0.085);
    g2.addColorStop(0, `rgb(${clamp(col2[0] * 255 + 40, 0, 255) | 0},${clamp(col2[1] * 255 + 40, 0, 255) | 0},${clamp(col2[2] * 255 + 40, 0, 255) | 0})`);
    g2.addColorStop(1, `rgb(${(col2[0] * 160) | 0},${(col2[1] * 160) | 0},${(col2[2] * 160) | 0})`);
    ctx.fillStyle = g2;
    ctx.beginPath();
    ctx.arc(p2x, p2y, scale * s * 0.085, 0, Math.PI * 2);
    ctx.fill();
    // Center star
    ctx.fillStyle = `rgba(255,255,200,0.9)`;
    ctx.shadowColor = "rgba(255,200,100,0.9)";
    ctx.shadowBlur = 12 * glow;
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  } else if (archetype === "particles") {
    // Particle field
    const num = 22;
    // deterministic pseudo-random per particle index + time
    for (let i = 0; i < num; i++) {
      const seed = i * 997 + Math.floor(tNorm * 60) % 4;
      const px = (Math.sin(tNorm * 1.8 + i * 0.7 + phase) * 0.9 + Math.cos(i * 0.31) * 0.5) * width * 0.45 + cx;
      const py = (Math.cos(tNorm * 1.2 + i * 1.1 + phase * 0.5) * 0.85 + Math.sin(i * 0.55) * 0.3) * height * 0.42 + cy;
      const sz = 1.2 + Math.sin(tNorm * 4 + i) * 0.8 + 1.5;
      const alpha = 0.5 + 0.4 * Math.sin(tNorm * 3 + i * 0.9);
      const col = i % 2 === 0 ? col1 : col2;
      ctx.globalAlpha = clamp(alpha, 0, 1) * (0.7 + glow * 0.3);
      // glow halo
      ctx.fillStyle = `rgba(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0},${0.15})`;
      ctx.beginPath();
      ctx.arc(px, py, sz * 2.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgb(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0})`;
      ctx.beginPath();
      ctx.arc(px, py, sz, 0, Math.PI * 2);
      ctx.fill();
      // sparkle
      if (i % 3 === 0) {
        ctx.strokeStyle = `rgba(255,255,255,${0.5 * alpha})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(px - sz * 0.8, py);
        ctx.lineTo(px + sz * 0.8, py);
        ctx.moveTo(px, py - sz * 0.8);
        ctx.lineTo(px, py + sz * 0.8);
        ctx.stroke();
      }
    }
  } else if (archetype === "gradient") {
    // Fluid gradient mesh
    // We already have bg, add advected blobs
    for (let i = 0; i < 3; i++) {
      const bx = cx + Math.cos(tNorm * 0.8 + i * 2.1 + phase) * width * 0.18;
      const by = cy + Math.sin(tNorm * 0.6 + i * 1.7) * height * 0.16;
      const br = s * (0.22 + Math.sin(tNorm * 1.2 + i) * 0.06) * (1 + glow * 0.25);
      const col = i % 2 === 0 ? col1 : col2;
      const grad2 = ctx.createRadialGradient(bx, by, 0, bx, by, br);
      grad2.addColorStop(0, `rgba(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0},${0.38})`);
      grad2.addColorStop(0.5, `rgba(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0},${0.14})`);
      grad2.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad2;
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fill();
    }
    // horizontal shift overlay
    ctx.globalAlpha = 0.07;
    ctx.fillStyle = `rgb(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0})`;
    ctx.fillRect(0, 0, width, height);
  } else if (archetype === "orbit2") {
    // Spiral vortex
    const loops = 16;
    ctx.translate(cx, cy);
    ctx.rotate(rotation * 0.6);
    for (let arm = 0; arm < 2; arm++) {
      ctx.save();
      ctx.rotate((arm * Math.PI));
      ctx.beginPath();
      for (let i = 0; i < loops; i++) {
        const t = i / loops;
        const ang = t * Math.PI * 3 + tNorm * 5 + phase;
        const rad = s * 0.04 + t * s * 0.32;
        const x = Math.cos(ang) * rad;
        const y = Math.sin(ang) * rad * 0.9;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      const col = arm === 0 ? col1 : col2;
      ctx.strokeStyle = `rgba(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0},${0.9})`;
      ctx.lineWidth = 2.2 + glow * 1.2;
      ctx.shadowColor = `rgba(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0},0.6)`;
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;
      // dots along spiral
      for (let i = 0; i < loops; i += 2) {
        const t = i / loops;
        const ang = t * Math.PI * 3 + tNorm * 5 + phase;
        const rad = s * 0.04 + t * s * 0.32;
        const x = Math.cos(ang) * rad;
        const y = Math.sin(ang) * rad * 0.9;
        ctx.globalAlpha = 0.9 - t * 0.5;
        ctx.fillStyle = `rgb(${(col[0] * 255) | 0},${(col[1] * 255) | 0},${(col[2] * 255) | 0})`;
        ctx.beginPath();
        ctx.arc(x, y, 2 + t * 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
    // center core
    ctx.globalAlpha = 1;
    const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, 12);
    coreGrad.addColorStop(0, `rgba(255,255,255,0.9)`);
    coreGrad.addColorStop(0.4, `rgba(${(col1[0] * 255) | 0},${(col1[1] * 255) | 0},${(col1[2] * 255) | 0},0.6)`);
    coreGrad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(0, 0, 16, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();

  // Subtle film grain overlay handled via CSS, but also subtle scanline
  // Keep canvas crisp
}

// Generate full video frames to data URLs + blob for download
export async function renderVideoToFrames(
  paramsSeq: FrameParams[],
  width: number,
  height: number,
  fps = 12
): Promise<{ frames: ImageData[]; canvases: HTMLCanvasElement[] }> {
  // This is stub - actual rendering happens in VideoPlayer component frame-by-frame via requestAnimationFrame
  return { frames: [], canvases: [] };
}
