"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { renderFrame } from "@/lib/renderer";
import type { FrameParams } from "@/lib/renderer";

interface Props {
  paramsSeq: FrameParams[];
  width?: number;
  height?: number;
  prompt: string;
  meta?: { archetype: string; confidence: number; genTime: number };
  onExport?: (blob: Blob) => void;
}

export default function VideoPlayer({ paramsSeq, width = 512, height = 512, prompt, meta }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(true);
  const [frameIdx, setFrameIdx] = useState(0);
  const [loop, setLoop] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);

  const numFrames = paramsSeq.length;
  const fps = 12;
  const duration = numFrames / fps;

  const drawFrame = useCallback(
    (idx: number, tOverall: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      // Use dpr for crispness
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = width * dpr;
      const h = height * dpr;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.scale(dpr, dpr);
      } else {
        // need to re-apply scale after resize? we already scaled, but subsequent draws keep transform
        // Instead we handle by saving/restoring: easier to reset transform each time
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      const params = paramsSeq[idx % numFrames];
      // tOverall is 0..duration, normalize to 0..1 for renderer
      const tNorm = (idx % numFrames) / Math.max(1, numFrames - 1);
      // Actually we want smooth t inside frame - we use global time for wave continuity
      const smoothT = tOverall / duration; // 0..1 loop
      renderFrame(ctx, params, smoothT, width, height);
    },
    [paramsSeq, width, height, numFrames]
  );

  // Animation loop
  useEffect(() => {
    if (!playing) return;
    let lastIdx = -1;
    const tick = (now: number) => {
      if (!startRef.current) startRef.current = now;
      const elapsed = (now - startRef.current) / 1000; // sec
      const looped = loop ? elapsed % duration : Math.min(elapsed, duration);
      const idx = Math.floor(looped * fps) % numFrames;
      const tOverall = looped;
      setCurrentTime(looped);
      if (idx !== lastIdx) {
        lastIdx = idx;
        setFrameIdx(idx);
        drawFrame(idx, tOverall);
      } else {
        // still need to draw for smooth sub-frame motion if renderer uses tOverall
        drawFrame(idx, tOverall);
      }
      if (!loop && elapsed >= duration) {
        setPlaying(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, loop, duration, fps, numFrames, drawFrame]);

  // When params change, redraw
  useEffect(() => {
    startRef.current = 0;
    setPlaying(true);
    setFrameIdx(0);
  }, [paramsSeq]);

  // Manual draw when paused and frame changes
  useEffect(() => {
    if (!playing) {
      const tOverall = (frameIdx / fps);
      drawFrame(frameIdx, tOverall);
    }
  }, [frameIdx, playing, drawFrame, fps]);

  const togglePlay = () => {
    if (playing) {
      setPlaying(false);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    } else {
      startRef.current = performance.now() - currentTime * 1000;
      setPlaying(true);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setFrameIdx(Math.floor(v * fps) % numFrames);
    setCurrentTime(v);
    startRef.current = performance.now() - v * 1000;
    drawFrame(Math.floor(v * fps) % numFrames, v / duration);
  };

  const handleExport = async () => {
    if (!canvasRef.current) return;
    setExporting(true);
    try {
      // Export as WebM via capturing frames -> stitch with canvas captureStream if available
      // Fallback: export as GIF-like sequence of PNGs in a zip? Simplify: export current canvas as PNG + also offer WebM via MediaRecorder
      const canvas = canvasRef.current;
      // Try MediaRecorder path for real video
      const stream = (canvas as any).captureStream?.(fps) ?? (canvas as any).mozCaptureStream?.(fps);
      if (stream && typeof MediaRecorder !== "undefined") {
        const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
        const chunks: BlobPart[] = [];
        recorder.ondataavailable = (e) => chunks.push(e.data);
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: "video/webm" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `lumen-${prompt.slice(0, 24).replace(/\s+/g, "-")}.webm`;
          a.click();
          setExporting(false);
        };
        recorder.start();
        // playback for duration
        const start = performance.now();
        // ensure playing
        setPlaying(true);
        startRef.current = performance.now();
        setTimeout(() => {
          recorder.stop();
          stream.getTracks().forEach((t: any) => t.stop());
        }, duration * 1000 + 200);
      } else {
        // Fallback PNG
        canvas.toBlob((blob) => {
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `lumen-${Date.now()}.png`;
          a.click();
          setExporting(false);
        }, "image/png");
      }
    } catch (e) {
      setExporting(false);
      // fallback
      console.error(e);
    }
  };

  return (
    <div className="group relative bg-[#0a0a0e] rounded-2xl overflow-hidden border border-zinc-800/80 shadow-[0_16px_48px_rgba(0,0,0,0.5)]">
      {/* Canvas */}
      <div className="relative aspect-square w-full bg-black overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          style={{ width: `${width}px`, height: `${height}px` }}
        />
        {/* Top bar */}
        <div className="absolute top-0 inset-x-0 p-3 flex items-center justify-between gap-2 bg-gradient-to-b from-black/60 to-transparent pointer-events-none">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono tracking-widest text-teal-300/90 bg-teal-500/10 border border-teal-400/20 px-2 py-1 rounded-full">
              LUMEN-0 • 52KB
            </span>
            {meta && (
              <span className="hidden sm:inline text-[10px] font-mono tracking-wide text-zinc-400 bg-zinc-800/60 border border-zinc-700/50 px-2 py-1 rounded-full">
                {meta.archetype} • {(meta.confidence * 100) | 0}% • {meta.genTime}ms
              </span>
            )}
          </div>
          <div className="flex items-center gap-1 pointer-events-auto">
            <button
              onClick={() => setLoop((v) => !v)}
              className={`text-[10px] font-mono px-2 py-1 rounded-full border transition ${loop ? "bg-white text-black border-white" : "bg-zinc-800 text-zinc-400 border-zinc-700 hover:text-zinc-200"}`}
            >
              LOOP {loop ? "ON" : "OFF"}
            </button>
          </div>
        </div>

        {/* Play overlay when paused */}
        {!playing && (
          <button
            onClick={togglePlay}
            className="absolute inset-0 grid place-items-center bg-black/20 backdrop-blur-[1px] transition"
          >
            <span className="w-14 h-14 rounded-full bg-white text-black grid place-items-center shadow-xl">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5.14v14l11-7-11-7z" />
              </svg>
            </span>
          </button>
        )}
        {/* Bottom seek */}
        <div className="absolute bottom-0 inset-x-0 p-3 bg-gradient-to-t from-black/70 to-transparent">
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlay}
              className="w-8 h-8 rounded-full bg-white text-black grid place-items-center shrink-0 hover:bg-zinc-100 transition"
            >
              {playing ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                </svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" className="ml-0.5">
                  <path d="M8 5.14v14l11-7-11-7z" />
                </svg>
              )}
            </button>
            <div className="flex-1 flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={duration}
                step={0.01}
                value={currentTime}
                onChange={handleSeek}
                className="w-full h-1 accent-white bg-zinc-700/60 rounded-full appearance-none cursor-pointer"
              />
              <span className="text-[10px] font-mono text-zinc-400 tabular-nums shrink-0">
                {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer controls */}
      <div className="px-3 py-2.5 flex items-center justify-between gap-2 bg-[#0f0f12] border-t border-zinc-800/60">
        <div className="flex items-center gap-2 text-[11px] font-mono text-zinc-500">
          <span className="hidden sm:inline">
            {width}×{height} • {fps}fps • {numFrames}f • {((numFrames * width * height * 3) / 1024 / 1024).toFixed(1)}MP
          </span>
          <span className="sm:hidden">
            {width}×{height} • {fps}fps
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExport}
            disabled={exporting}
            className="text-xs font-medium px-3 py-1.5 rounded-full bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-200 transition disabled:opacity-50 flex items-center gap-1.5"
          >
            {exporting ? (
              <>
                <span className="w-3 h-3 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" />
                Exporting…
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Export WebM
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
