import {
  browserEngine,
  type ChatMessage,
  type ChatResponse,
} from "./engine/glassboxBrowser";
import type {
  AttentionAnalysis,
  AutogradDagResponse,
  BPEInspectResponse,
  EngineVerifyResponse,
  GeneratedTokenStep,
  GenerateResponse,
  LabStatus,
  LogitLensAnalysis,
  KVCacheBenchResult,
  LLM100MProof,
  MatmulBenchRow,
  PatchingAnalysis,
  StatusResponse,
  StepRecord,
} from "./types";

export type RuntimeMode = "server" | "browser";

let detectedMode: RuntimeMode | null = null;
let detectPromise: Promise<RuntimeMode> | null = null;

export async function detectRuntimeMode(): Promise<RuntimeMode> {
  if (detectedMode) return detectedMode;
  if (detectPromise) return detectPromise;

  detectPromise = (async () => {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1400);
      const res = await fetch("/api/health", { signal: ctrl.signal });
      clearTimeout(timer);
      if (res.ok) {
        const ct = res.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const data = await res.json();
          if (data && data.ok === true) {
            detectedMode = "server";
            return "server";
          }
        }
      }
    } catch {
      // Static Vercel deployment — fall back cleanly to In-Browser Float32 Engine
    }
    await browserEngine.init("chat");
    detectedMode = "browser";
    return "browser";
  })();

  return detectPromise;
}

export function getRuntimeMode(): RuntimeMode {
  return detectedMode ?? "browser";
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.error || `HTTP ${res.status}`);
  }
  return data as T;
}

export const api = {
  detectMode: detectRuntimeMode,

  getStatus: async (): Promise<StatusResponse> => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.getStatus();
    }
    return req<StatusResponse>("/api/status");
  },

  switchModel: async (
    modelId: "chat" | "induction"
  ): Promise<{ status: LabStatus; history: StepRecord[] }> => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      return browserEngine.switchModel(modelId);
    }
    return req<{ status: LabStatus; history: StepRecord[] }>("/api/models/switch", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
  },

  /** Live streaming chat generator — true token-by-token neural inference in browser or server! */
  chatStream: async function* (payload: {
    messages: ChatMessage[];
    max_new_tokens?: number;
    temperature?: number;
    top_k?: number;
    top_p?: number;
    ablate_heads?: [number, number][];
    ablation_mode?: "zero" | "mean";
  }): AsyncGenerator<
    { prompt: string; token: GeneratedTokenStep; partialReply: string },
    ChatResponse,
    void
  > {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return yield* browserEngine.chatStream(payload);
    }
    // Server mode: call /api/chat and animate token reveal smoothly
    const res = await req<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    let acc = "";
    for (const tok of res.tokens) {
      acc += tok.text;
      if (acc.includes("\nUser:")) break;
      yield { prompt: res.prompt, token: tok, partialReply: acc.trimStart() };
      await new Promise((r) => setTimeout(r, 12));
    }
    return res;
  },

  startTraining: async (steps: number, lr?: number) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      throw new Error(
        "Live C++17 training runs on the local Python backend (`python3 server/app.py`). On Vercel, inference, Chat, Attention Ablation, Logit Lens, DLA, Causal Patching, and BPE run live in-browser!"
      );
    }
    return req<{ status: LabStatus }>("/api/train/start", {
      method: "POST",
      body: JSON.stringify({ steps, lr }),
    });
  },

  stopTraining: async () => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      return { status: browserEngine.getStatus().status };
    }
    return req<{ status: LabStatus }>("/api/train/stop", { method: "POST", body: "{}" });
  },

  trainOneStep: async () => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      throw new Error(
        "Live gradient steps require the local C++17 Python server (`python3 server/app.py`). All inference & mechanistic interpretability tools run natively in your browser!"
      );
    }
    return req<{ step_record: StepRecord; status: LabStatus }>("/api/train/step", {
      method: "POST",
      body: "{}",
    });
  },

  resetSession: async (payload: {
    dataset_id: string;
    preset_id: string;
    custom_text?: string;
    config_overrides?: Record<string, number>;
    lr: number;
    batch_size: number;
    weight_decay?: number;
    grad_clip?: number;
    seed?: number;
    warm_steps?: number;
  }) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      const target = payload.dataset_id === "induction" ? "induction" : "chat";
      return browserEngine.switchModel(target);
    }
    return req<{ status: LabStatus; history: StepRecord[] }>("/api/session/reset", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  generate: async (payload: {
    prompt: string;
    max_new_tokens: number;
    temperature: number;
    top_k: number;
    top_p: number;
  }) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.generate(payload);
    }
    return req<GenerateResponse>("/api/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  analyzeAttention: async (payload: {
    prompt: string;
    ablate_heads?: [number, number][];
    ablation_mode?: string;
  }) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.analyzeAttention(payload);
    }
    return req<AttentionAnalysis>("/api/interp/attention", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  analyzeLogitLens: async (payload: {
    prompt: string;
    target_token?: string;
    foil_token?: string;
  }) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.analyzeLogitLens(payload);
    }
    return req<LogitLensAnalysis>("/api/interp/logit-lens", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  runPatching: async (payload: {
    clean_prompt: string;
    corrupted_prompt: string;
    clean_target?: string;
    corrupted_target?: string;
  }) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.runPatching(payload);
    }
    return req<PatchingAnalysis>("/api/interp/patching", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  inspectBPE: async (text: string) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.inspectBPE(text);
    }
    return req<BPEInspectResponse>("/api/bpe/inspect", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  },

  captureAutogradDag: async (prompt?: string) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.getAutogradDagSnapshot();
    }
    return req<AutogradDagResponse>("/api/engine/autograd-dag", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    });
  },

  runEngineVerify: async () => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.getVerifySnapshot();
    }
    return req<EngineVerifyResponse>("/api/engine/verify", {
      method: "POST",
      body: "{}",
    });
  },

  runMatmulBenchmark: async (sizes: number[] = [64, 128, 256]) => {
    const mode = await detectRuntimeMode();
    if (mode === "browser") {
      await browserEngine.init(browserEngine.getActiveModelId());
      return browserEngine.runBrowserMatmulBenchmark(sizes.filter((s) => s <= 256));
    }
    return req<{ results: MatmulBenchRow[] }>("/api/engine/benchmark", {
      method: "POST",
      body: JSON.stringify({ sizes }),
    });
  },

  fetch100MProof: async (): Promise<LLM100MProof> => {
    const res = await fetch("/models/glassbox-100m-proof.json");
    if (!res.ok) throw new Error(`HTTP ${res.status} loading Glassbox-100M proof`);
    return (await res.json()) as LLM100MProof;
  },

  runKVCacheBenchmark: async (): Promise<KVCacheBenchResult> => {
    await browserEngine.init("chat");
    return browserEngine.benchmarkKVCache(28, 24);
  },
};

export type { ChatMessage, ChatResponse };
