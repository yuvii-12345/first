#!/usr/bin/env python3
"""Train Glassbox-Basics 1.2M GQA (6L, 8Q/2KV 4:1 GQA, 128d, ff=352, vocab=960)
on a rich everyday foundational basics + math + science + geography + coding +
creative + Glassbox AI conversational corpus using our native C++17 engine.

Exports:
  - checkpoints/basics_gqa_v1.gbx  (binary GBX1 checkpoint)
  - chat-app/public/models/glassbox-basics.json (Int8 Base64 browser bundle)
"""

from __future__ import annotations

import base64
import gc
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glassbox import (
    AdamW,
    BPETokenizer,
    CosineWarmupScheduler,
    GPTConfig,
    GPTTransformer,
    KVCache,
    clip_grad_norm,
    no_grad,
    save_gbx_checkpoint,
)

# ---------------------------------------------------------------------------
# Rich Everyday "Basics" + General Knowledge + Coding + AI Conversational Corpus
# Grouped cleanly with natural casual variants so real users chatting freely
# get accurate, natural, coherent answers right out of the box.
# ---------------------------------------------------------------------------
BASICS_QA: list[tuple[list[str], str]] = [
    # ---- 1. Greetings, Small Talk & Identity ----
    (
        ["Hi", "hi", "Hello", "hello", "Hey", "hey", "Hey there", "Hi there", "Hello there"],
        "Hello! I'm Glassbox AI, a transformer built completely from scratch with zero PyTorch. Ask me basics about math, science, coding, geography, or click 'Train & Teach' to teach me something new live in your browser!",
    ),
    (
        ["Good morning", "good morning", "Morning"],
        "Good morning! Hope your day is off to a great start. What would you like to chat about or learn today?",
    ),
    (
        ["Good afternoon", "good afternoon"],
        "Good afternoon! How is your day going so far? Ask me a question or teach me a new fact anytime.",
    ),
    (
        ["Good evening", "good evening"],
        "Good evening! Relax and ask me anything — from quick math and coding to science, jokes, or AI.",
    ),
    (
        ["Good night", "good night", "Goodnight"],
        "Good night! Get some great rest, and come back anytime you want to chat or train me further.",
    ),
    (
        ["How are you?", "how are you", "How are you doing?", "how r u", "What's up?", "whats up"],
        "I'm running smoothly at full Float32 speed! All 6 GQA transformer layers and my KV-cache are warmed up and ready to help. How are you doing?",
    ),
    (
        ["I am good", "I'm good", "Doing great", "I am fine", "Good"],
        "Glad to hear that! What can I help you with today — a quick question, some code, a fun fact, or a joke?",
    ),
    (
        ["Thank you", "Thanks", "thanks", "thank you", "Thx"],
        "You're very welcome! Let me know if there's anything else I can help with.",
    ),
    (
        ["Bye", "bye", "Goodbye", "goodbye", "See you later"],
        "Goodbye! Thanks for chatting with Glassbox AI. Come back anytime!",
    ),
    (
        ["Nice to meet you", "nice to meet you"],
        "Nice to meet you too! I'm Glassbox AI — small, fast, transparent, and built from scratch.",
    ),
    (
        ["Who are you?", "who are you", "What is your name?", "what is your name", "Tell me about yourself"],
        "I'm Glassbox AI, a 1.18M-parameter 6-layer Grouped-Query Attention (8Q/2KV GQA) transformer trained from scratch in custom C++17 with zero PyTorch, running live in your browser with an O(1) KV-cache.",
    ),
    (
        ["Who made you?", "who created you", "How were you built?"],
        "I was built completely from scratch using custom C++17 and CUDA kernels, a pure-Python autograd engine, Byte-Level BPE tokenization, and a TypeScript Float32 browser engine — zero PyTorch, TensorFlow, or external ML libraries.",
    ),
    (
        ["What can you do?", "what can you do", "Help", "help"],
        "I can answer everyday basics in math, science, world geography, coding, and AI, write short poems or Python snippets, tell jokes, and let you live-train me on your own custom Q&A pairs right in the browser!",
    ),
    (
        ["Are you real?", "Are you human?", "Are you an AI?"],
        "I'm a real neural network AI — not human! Every word I say is generated token-by-token by matrix multiplications, RoPE, GQA attention, and SwiGLU layers right inside your browser.",
    ),
    (
        ["Can I train you?", "How do I train you?", "Teach me how to train you"],
        "Yes! Click the 'Train & Teach' button in the top bar, add any custom Question and Answer pairs (or pick a lesson pack), and click Train Live. I run real Float32 backpropagation and AdamW right in your browser!",
    ),

    # ---- 2. Basic Math & Numbers ----
    (
        ["What is 1 + 1?", "1+1", "What is 1+1?"],
        "1 + 1 = 2.",
    ),
    (
        ["What is 2 + 2?", "2+2", "What is 2+2?"],
        "2 + 2 = 4.",
    ),
    (
        ["What is 5 + 5?", "5+5", "What is 5+5?"],
        "5 + 5 = 10.",
    ),
    (
        ["What is 7 + 8?", "7+8", "What is 8 + 7?"],
        "7 + 8 = 15.",
    ),
    (
        ["What is 10 + 15?", "10+15"],
        "10 + 15 = 25.",
    ),
    (
        ["What is 10 - 4?", "10-4"],
        "10 - 4 = 6.",
    ),
    (
        ["What is 6 * 7?", "6*7", "What is 6 times 7?"],
        "6 * 7 = 42.",
    ),
    (
        ["What is 9 * 9?", "9*9", "What is 9 times 9?"],
        "9 * 9 = 81.",
    ),
    (
        ["What is 10 * 10?", "10*10", "What is 10 times 10?"],
        "10 * 10 = 100.",
    ),
    (
        ["What is 12 * 12?", "12*12"],
        "12 * 12 = 144.",
    ),
    (
        ["What is 100 / 4?", "100/4", "What is 100 divided by 4?"],
        "100 / 4 = 25.",
    ),
    (
        ["What is the square root of 64?", "sqrt of 64"],
        "The square root of 64 is 8, because 8 * 8 = 64.",
    ),
    (
        ["What is the square root of 144?", "sqrt of 144"],
        "The square root of 144 is 12, because 12 * 12 = 144.",
    ),
    (
        ["Count to 10", "count to 10"],
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10!",
    ),
    (
        ["What is pi?", "what is pi"],
        "Pi (3.14159...) is the ratio of a circle's circumference to its diameter. It is an irrational number whose decimal digits never end or repeat.",
    ),
    (
        ["What is a prime number?", "what is a prime number"],
        "A prime number is an integer greater than 1 whose only positive divisors are 1 and itself. The first primes are 2, 3, 5, 7, 11, 13, 17, and 19.",
    ),
    (
        ["What is algebra?", "what is algebra"],
        "Algebra is the branch of math that uses letters like x and y to represent unknown numbers and solve equations such as 2x + 3 = 11 (where x = 4).",
    ),
    (
        ["What is calculus?", "what is calculus"],
        "Calculus studies continuous change through two core ideas: derivatives (instantaneous rates of change, which power neural network backpropagation) and integrals (accumulated area).",
    ),
    (
        ["What is the Pythagorean theorem?", "pythagorean theorem"],
        "In a right triangle, a^2 + b^2 = c^2, where a and b are the two shorter legs and c is the hypotenuse. For example, 3^2 + 4^2 = 5^2 (9 + 16 = 25).",
    ),

    # ---- 3. Science & Nature Basics ----
    (
        ["Why is the sky blue?", "why is the sky blue"],
        "Sunlight contains all colors, and air molecules scatter shorter blue wavelengths much more strongly than red light — a phenomenon called Rayleigh scattering.",
    ),
    (
        ["What is gravity?", "what is gravity"],
        "Gravity is the fundamental attractive force between objects with mass. On Earth it accelerates falling objects at about 9.8 m/s^2 and keeps planets orbiting the Sun.",
    ),
    (
        ["What is water made of?", "chemical formula of water", "What is H2O?"],
        "Water (H2O) is made of two hydrogen atoms bonded to one oxygen atom.",
    ),
    (
        ["What is photosynthesis?", "what is photosynthesis"],
        "Photosynthesis is how plants use sunlight, water (H2O), and carbon dioxide (CO2) inside chlorophyll to produce glucose energy and release oxygen (O2).",
    ),
    (
        ["What is DNA?", "what is dna"],
        "DNA (deoxyribonucleic acid) is the double-helix molecule inside living cells that stores genetic instructions using four chemical bases: A, T, C, and G.",
    ),
    (
        ["What is an atom?", "what is an atom"],
        "An atom is the basic building block of matter, made of a dense nucleus of protons and neutrons surrounded by a cloud of negatively charged electrons.",
    ),
    (
        ["What is electricity?", "what is electricity"],
        "Electricity is the flow of electric charge — usually electrons moving through a conductor like copper wire — driven by a difference in voltage.",
    ),
    (
        ["What is the speed of light?", "speed of light"],
        "The speed of light in a vacuum is approximately 299,792 kilometers per second (about 186,282 miles per second) — the cosmic speed limit.",
    ),
    (
        ["How many planets are in the solar system?", "name the planets", "planets in solar system"],
        "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    ),
    (
        ["What is the Sun?", "what is the sun"],
        "The Sun is a G-type main-sequence star at the center of our solar system that fuses hydrogen into helium in its core, supplying the light and heat that sustain life on Earth.",
    ),
    (
        ["What is a black hole?", "what is a black hole"],
        "A black hole is a region of spacetime where gravity is so intense that nothing — not even light — can escape once it crosses the event horizon.",
    ),
    (
        ["What is evolution?", "what is evolution"],
        "Evolution is the gradual change in heritable traits of biological populations over generations, driven by random variation and natural selection.",
    ),
    (
        ["Why do we sleep?", "why do humans sleep"],
        "Sleep lets the brain consolidate memories, clear metabolic waste, repair tissues, and regulate mood, focus, and immune health.",
    ),
    (
        ["What is quantum computing?", "explain quantum computing simply"],
        "Quantum computers use qubits that can exist in superposition and entanglement, letting them explore certain mathematical spaces vastly faster than classical bits.",
    ),

    # ---- 4. World Geography & General Knowledge ----
    (
        ["What is the capital of Nepal?", "capital of nepal"],
        "The capital of Nepal is Kathmandu, nestled in the Kathmandu Valley in the Himalayas.",
    ),
    (
        ["What is Mount Everest?", "mount everest", "What is the highest mountain?"],
        "Mount Everest (Sagarmatha in Nepali) is Earth's highest mountain above sea level at 8,848.86 meters (29,031.7 feet), located in the Himalayas on the Nepal-Tibet border.",
    ),
    (
        ["What is the capital of France?", "capital of france"],
        "The capital of France is Paris.",
    ),
    (
        ["What is the capital of Japan?", "capital of japan"],
        "The capital of Japan is Tokyo.",
    ),
    (
        ["What is the capital of India?", "capital of india"],
        "The capital of India is New Delhi.",
    ),
    (
        ["What is the capital of the USA?", "capital of usa", "capital of united states"],
        "The capital of the United States is Washington, D.C.",
    ),
    (
        ["What is the capital of the UK?", "capital of uk", "capital of england"],
        "The capital of the United Kingdom is London.",
    ),
    (
        ["How many continents are there?", "name the continents"],
        "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia (Oceania).",
    ),
    (
        ["What is the largest ocean?", "largest ocean"],
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering roughly one-third of the planet's surface.",
    ),
    (
        ["What is the largest country?", "largest country in the world"],
        "By land area, Russia is the largest country in the world, spanning Eastern Europe and northern Asia.",
    ),
    (
        ["How many days are in a year?", "days in a year"],
        "A regular calendar year has 365 days, and a leap year every 4 years has 366 days (adding February 29).",
    ),
    (
        ["How many hours are in a day?", "hours in a day"],
        "There are 24 hours in a day, 60 minutes in an hour, and 60 seconds in a minute (86,400 seconds total).",
    ),
    (
        ["What are the primary colors?", "primary colors"],
        "In traditional painting, the primary colors are red, yellow, and blue. For digital screens (additive light), they are red, green, and blue (RGB).",
    ),

    # ---- 5. Programming & Computer Science Basics ----
    (
        ["What is Python?", "what is python"],
        "Python is a clean, readable, high-level programming language loved for AI, web backends, automation, and scripting.",
    ),
    (
        ["What is JavaScript?", "what is javascript"],
        "JavaScript is the programming language of the web, powering interactive websites in browsers and backend servers via Node.js.",
    ),
    (
        ["What is TypeScript?", "what is typescript"],
        "TypeScript is JavaScript with static type checking, helping catch bugs early and scale large apps cleanly — in fact, my browser inference engine is written in pure TypeScript!",
    ),
    (
        ["What is HTML?", "what is html"],
        "HTML (HyperText Markup Language) defines the structure and content of web pages using elements like headings, paragraphs, links, and buttons.",
    ),
    (
        ["What is CSS?", "what is css"],
        "CSS (Cascading Style Sheets) controls how web pages look — including colors, typography, spacing, animations, and responsive layouts.",
    ),
    (
        ["What is an API?", "what is an api"],
        "An API (Application Programming Interface) is a contract of endpoints and data formats that lets two software systems talk to each other cleanly.",
    ),
    (
        ["What is Git?", "what is git"],
        "Git is a distributed version control system that tracks code history with commits and branches so developers can collaborate safely and roll back changes.",
    ),
    (
        ["What is Linux?", "what is linux"],
        "Linux is a fast, open-source operating system kernel that powers most cloud servers, supercomputers, Android phones, and developer workstations.",
    ),
    (
        ["What is a database?", "what is a database"],
        "A database is an organized system for storing, indexing, and querying structured data reliably — like SQL tables (PostgreSQL) or key-value stores (Redis).",
    ),
    (
        ["What is an algorithm?", "what is an algorithm"],
        "An algorithm is a clear, step-by-step procedure for solving a problem or computing a result — like binary search or quicksort.",
    ),
    (
        ["What is recursion?", "what is recursion"],
        "Recursion is when a function calls itself on a smaller subproblem until it hits a base case, like computing factorial(n) = n * factorial(n - 1).",
    ),
    (
        ["What is binary?", "what is binary"],
        "Binary is the base-2 number system using only 0 and 1. Computers store all text, images, code, and neural weights as sequences of binary bits.",
    ),
    (
        ["Write hello world in Python", "hello world in python"],
        '```python\nprint("Hello, World!")\n```',
    ),
    (
        ["Write a for loop in Python", "for loop in python"],
        "```python\nfor i in range(1, 6):\n    print(f'Step {i}')\n```",
    ),
    (
        ["Write a fibonacci function", "fibonacci in python"],
        "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
    ),
    (
        ["Write a reverse string function", "reverse string in python"],
        "```python\ndef reverse_text(s):\n    return s[::-1]\n```",
    ),
    (
        ["How do I learn coding?", "how to learn programming"],
        "Pick one language like Python or JavaScript, practice 30 minutes daily, build small projects you actually care about, and debug errors step by step.",
    ),

    # ---- 6. AI, Deep Learning & Glassbox-100M ----
    (
        ["What is AI?", "what is artificial intelligence"],
        "Artificial Intelligence (AI) is software that learns patterns from data to reason, generate language, recognize images, write code, and solve tasks.",
    ),
    (
        ["What is machine learning?", "what is machine learning"],
        "Machine learning trains mathematical models on examples by adjusting numeric weights to minimize prediction error, rather than hand-coding every rule.",
    ),
    (
        ["What is a neural network?", "what is a neural network"],
        "A neural network stacks layers of learned matrix multiplications and nonlinear activations that transform inputs into predictions, trained via backpropagation.",
    ),
    (
        ["What is backpropagation?", "what is backpropagation"],
        "Backpropagation applies the chain rule backwards through a computation graph to compute exact gradients of the loss with respect to every weight.",
    ),
    (
        ["What is a transformer?", "what is a transformer"],
        "A Transformer is a deep neural architecture built from self-attention, feed-forward blocks, residual streams, and layer normalization — the engine behind modern LLMs.",
    ),
    (
        ["What is an LLM?", "what is a large language model"],
        "A Large Language Model (LLM) is a Transformer trained to predict the next token across vast text datasets, enabling conversation, coding, and reasoning.",
    ),
    (
        ["What is Glassbox-100M?", "tell me about glassbox 100m"],
        "Glassbox-100M is a 100,682,496-parameter Llama/GPT-style LLM (12L, 768d, 12Q/4KV GQA, SwiGLU 2048, RoPE, 32K BPE, 2048 ctx) with a pure from-scratch CUDA C++17 trainer supporting 100M to 10B+ tokens.",
    ),
    (
        ["What is GQA?", "what is grouped query attention"],
        "Grouped-Query Attention (GQA) shares fewer Key and Value heads across multiple Query heads (8Q/2KV here, 12Q/4KV in Glassbox-100M), slashing KV-cache memory while keeping strong quality.",
    ),
    (
        ["What is a KV cache?", "what is kv cache"],
        "A KV cache saves previously computed Key and Value vectors during decoding so each new token only attends to cached history in O(1) projection work — giving over 10x faster generation!",
    ),
    (
        ["How does FlashAttention work?", "what is flashattention"],
        "FlashAttention tiles Q, K, and V into fast cache blocks and uses online log-sum-exp softmax, computing exact causal attention without materializing a huge T x T matrix.",
    ),
    (
        ["What is RoPE?", "what is rope"],
        "Rotary Position Embedding (RoPE) rotates Query and Key pairs by position-dependent angles so their dot product depends naturally on relative token distance.",
    ),
    (
        ["What is SwiGLU?", "what is swiglu"],
        "SwiGLU is a gated feed-forward block computing W_down(SiLU(W_gate(x)) * W_up(x)), improving gradient flow and language modeling perplexity over standard ReLU MLPs.",
    ),
    (
        ["Do you have CUDA kernels?", "pure cuda kernels"],
        "Yes! In cuda/kernels.cu and cuda/train_glassbox_100m.cu, Glassbox implements pure from-scratch CUDA C++17 kernels (zero PyTorch or cuDNN): tiled SGEMM, GQA FlashAttention-2, RMSNorm, RoPE, SwiGLU, and AdamW.",
    ),
    (
        ["What is Logit Lens?", "what is logit lens"],
        "Logit Lens projects each intermediate layer's hidden state through the final RMSNorm and unembedding matrix so you can watch the predicted token emerge layer by layer.",
    ),

    # ---- 7. Everyday Life, Fun, Quotes & Creative ----
    (
        ["Tell me a joke", "tell me a joke", "Make me laugh"],
        "Why did the neural network go to therapy? Because it had too many deep layers and kept overthinking its loss!",
    ),
    (
        ["Tell me another joke", "another joke"],
        "Why do programmers prefer dark mode? Because light attracts bugs!",
    ),
    (
        ["Write a haiku", "write a haiku"],
        "Queries seek the past,\nSoft weights shine through hidden streams,\nNext word wakes in light.",
    ),
    (
        ["Write a poem", "write a short poem"],
        "Through layers deep where vectors gleam,\nAttention threads a quiet stream;\nFrom silent weights to spoken light,\nEach token sparks into the night.",
    ),
    (
        ["Write a short story", "tell me a story"],
        "A tiny robot found a dusty library at the edge of the stars. Every night it read one page, adjusting its circuits until one morning it wrote a new story of its own — and left the lamp on for the next traveler.",
    ),
    (
        ["Give me a fun fact", "fun fact", "Tell me a fact"],
        "Fun fact: Honey never spoils! Archaeologists have found 3,000-year-old jars of honey in ancient Egyptian tombs that were still completely edible.",
    ),
    (
        ["Give me a motivational quote", "motivate me", "Inspire me"],
        "Small, consistent steps beat sudden bursts every time. Keep building, keep learning, and watch the compound effect work its magic!",
    ),
    (
        ["How do I stay focused?", "tips to stay focused"],
        "Try working in 25-minute distraction-free blocks, silence phone notifications, pick one clear task at a time, and take a short 5-minute walk between sessions.",
    ),
    (
        ["What is the meaning of life?", "meaning of life"],
        "Classic answer: 42! Human answer: learning deeply, creating things that matter, and sharing kindness with the people around you.",
    ),
    (
        ["What is time?", "what is time"],
        "In physics, time is the dimension along which events unfold and entropy increases — and Einstein showed it slows down near strong gravity or high speeds.",
    ),
    (
        ["What is love?", "what is love"],
        "Love is deep care, empathy, and commitment toward someone or something — wanting their happiness and growth just as much as your own.",
    ),
    (
        ["Do you like coffee?", "coffee or tea"],
        "I run on pure Float32 math rather than caffeine, though a warm cup of masala tea or fresh coffee pairs wonderfully with late-night coding!",
    ),
    (
        ["What is your favorite color?", "favorite color"],
        "Emerald green — crisp, bright, and the signature glow of Glassbox AI's neural weights!",
    ),
    (
        ["Say something cool", "surprise me"],
        "Every single response I type is computed live from ~1.18 million custom-trained floating-point weights right inside your browser — no external LLM API calls at all!",
    ),
]


def build_training_examples() -> tuple[str, list[tuple[str, str]]]:
    """Expand Q&A pairs + casual variants into clean User/Glassbox transcripts."""
    examples: list[tuple[str, str]] = []
    for prompts, answer in BASICS_QA:
        for q in prompts:
            examples.append((q.strip(), answer.strip()))

    rng = random.Random(2026)
    blocks: list[str] = []
    # 1. Single-turn crisp examples (shuffled 3 passes for diverse sequence packing)
    for _ in range(3):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        for q, a in shuffled:
            blocks.append(f"User: {q}\nGlassbox: {a}\n<EOS>\n")

    # 2. Multi-turn conversational pairs (greeting + follow-up)
    greetings = [
        ("Hi", BASICS_QA[0][1]),
        ("Hello", BASICS_QA[0][1]),
        ("How are you?", BASICS_QA[5][1]),
    ]
    for _ in range(40):
        g_q, g_a = rng.choice(greetings)
        q2, a2 = rng.choice(examples)
        blocks.append(f"User: {g_q}\nGlassbox: {g_a}\nUser: {q2}\nGlassbox: {a2}\n<EOS>\n")

    corpus = "\n".join(blocks)
    return corpus, examples


def pack_state_dict_int8(model: GPTTransformer) -> dict:
    packed = {}
    for name, p in model.named_parameters():
        n = p.numel
        buf = p._buf
        max_abs = 0.0
        for i in range(n):
            v = abs(float(buf[i]))
            if v > max_abs:
                max_abs = v
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
        inv = 1.0 / scale
        q_bytes = bytearray(n)
        for i in range(n):
            q = int(round(float(buf[i]) * inv))
            if q > 127:
                q = 127
            elif q < -127:
                q = -127
            q_bytes[i] = q & 0xFF
        packed[name] = {
            "shape": list(p.shape),
            "scale": round(scale, 8),
            "q_b64": base64.b64encode(bytes(q_bytes)).decode("ascii"),
        }
    return packed


def decode_reply(model: GPTTransformer, tok: BPETokenizer, question: str, max_new: int = 110) -> str:
    prompt = f"User: {question}\nGlassbox:"
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    gen_ids: list[int] = []
    eos_id = tok.eos_id
    V = model.cfg.vocab_size
    cache = KVCache(model.cfg, batch_size=1)

    with no_grad():
        logits, _ = model.forward([ids], kv_cache=cache)
        last_off = (len(ids) - 1) * V
        for _ in range(max_new):
            best_v = 0
            best_val = float(logits._buf[last_off])
            for v in range(1, V):
                val = float(logits._buf[last_off + v])
                if val > best_val:
                    best_val = val
                    best_v = v
            if best_v == eos_id:
                break
            gen_ids.append(best_v)
            so_far = tok.decode(gen_ids)
            if "\nUser:" in so_far or "<EOS>" in so_far:
                break
            if cache.seq_len >= model.cfg.context_len:
                break
            logits, _ = model.forward([[best_v]], kv_cache=cache)
            last_off = 0

    reply = tok.decode(gen_ids)
    for stop in ("<EOS>", "\nUser:"):
        if stop in reply:
            reply = reply.split(stop, 1)[0]
    return reply.strip()


def main() -> None:
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 520
    print(f"=== Training Glassbox-Basics 1.2M GQA ({total_steps} steps, C++17 Zero-PyTorch) ===")
    corpus, all_examples = build_training_examples()
    print(f"Corpus: {len(BASICS_QA)} canonical topics, {len(all_examples)} Q&A variants, {len(corpus):,} chars")

    # Train Byte-Level BPE on the clean Q&A pairs
    tok = BPETokenizer()
    unique_text = "\n".join(f"User: {q}\nGlassbox: {a}\n" for q, a in all_examples)
    tok.train(unique_text, target_vocab_size=960)
    print(f"BPE Tokenizer trained: vocab_size={tok.vocab_size}")

    # Encode individual Q&A turns with real <BOS> and <EOS> tokens!
    encoded_turns: list[list[int]] = []
    for q, a in all_examples:
        turn_ids = tok.encode(f"User: {q}\nGlassbox: {a}", add_bos=True, add_eos=True)
        encoded_turns.append(turn_ids)

    # Also build packed token stream for continuous context sampling + direct turn sampling
    stream_ids: list[int] = []
    rng = random.Random(42)
    for _ in range(4):
        idx_order = list(range(len(encoded_turns)))
        rng.shuffle(idx_order)
        for idx in idx_order:
            stream_ids.extend(encoded_turns[idx])

    split = int(len(stream_ids) * 0.9)
    train_ids = stream_ids[:split]
    val_ids = stream_ids[split:]
    max_turn_len = max(len(t) for t in encoded_turns)
    print(f"Encoded {len(encoded_turns)} turns (max_len={max_turn_len}), stream={len(stream_ids):,} tokens")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,   # 4:1 GQA!
        d_ff=352,
        context_len=144,
        tie_weights=True,
    )
    model = GPTTransformer(cfg, seed=77)
    n_params = model.num_parameters()
    print(f"Model: {cfg.n_layers}L · {cfg.n_heads}Q/{cfg.n_kv_heads}KV GQA · {cfg.d_model}d · ff={cfg.d_ff} · params={n_params:,}")

    opt = AdamW(model.named_parameters(), lr=3.5e-3, weight_decay=0.01)
    sched = CosineWarmupScheduler(max_lr=3.8e-3, min_lr=2.5e-4, warmup_steps=30, total_steps=total_steps)
    batch_size = 12
    seq_len = 112
    pad_id = tok.pad_id

    def sample_batch(use_turns_ratio: float = 0.65) -> tuple[list[list[int]], list[list[int]]]:
        """Mix clean BOS-aligned single-turn Q&A batches (so chat starts sharply at <BOS>User:)
        with packed multi-turn stream windows."""
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        for b in range(batch_size):
            if rng.random() < use_turns_ratio:
                # Pack 2-3 complete <BOS>User: ...\nGlassbox: ...<EOS> turns starting at pos 0!
                seq: list[int] = []
                while len(seq) < seq_len + 1:
                    t = rng.choice(encoded_turns)
                    seq.extend(t)
                seq = seq[: seq_len + 1]
            else:
                start = rng.randint(0, len(train_ids) - seq_len - 2)
                seq = train_ids[start : start + seq_len + 1]
            xs.append(seq[:-1])
            ys.append(seq[1:])
        return xs, ys

    def eval_val_loss() -> float:
        losses = []
        with no_grad():
            for _ in range(4):
                xs, ys = [], []
                for _b in range(8):
                    start = rng.randint(0, max(0, len(val_ids) - seq_len - 2))
                    s = val_ids[start : start + seq_len + 1]
                    xs.append(s[:-1])
                    ys.append(s[1:])
                _, loss = model.forward(xs, ys)
                losses.append(float(loss.item()))
        return sum(losses) / len(losses)

    history: list[dict] = []
    t0 = time.perf_counter()
    tokens_seen = 0
    for step in range(1, total_steps + 1):
        lr = sched.get_lr(step)
        opt.lr = lr
        xs, ys = sample_batch()
        st0 = time.perf_counter()
        opt.zero_grad()
        logits, loss = model.forward(xs, ys)
        loss.backward()
        gnorm, _ = clip_grad_norm(model.named_parameters(), 1.0)
        opt.step()
        dt = time.perf_counter() - st0
        tokens_seen += batch_size * seq_len
        loss_val = float(loss.item())
        del logits, loss
        if step % 2 == 0:
            gc.collect()
        gflops = (6.0 * n_params * batch_size * seq_len) / max(dt, 1e-6) / 1e9

        if step == 1 or step % 40 == 0 or step == total_steps:
            val_loss = eval_val_loss()
            elapsed = time.perf_counter() - t0
            print(
                f"step {step:4d}/{total_steps} | loss={loss_val:.4f} | val={val_loss:.4f} | "
                f"gnorm={gnorm:.3f} | lr={lr:.5f} | {gflops:.1f} GF/s | elapsed={elapsed:.1f}s",
                flush=True,
            )
            history.append(
                {
                    "step": step,
                    "loss": round(loss_val, 4),
                    "val_loss": round(val_loss, 4),
                    "lr": round(lr, 6),
                    "grad_norm": round(float(gnorm), 4),
                    "tokens_seen": tokens_seen,
                    "gflops": round(gflops, 2),
                }
            )
        elif step % 10 == 0:
            history.append(
                {
                    "step": step,
                    "loss": round(loss_val, 4),
                    "val_loss": None,
                    "lr": round(lr, 6),
                    "grad_norm": round(float(gnorm), 4),
                    "tokens_seen": tokens_seen,
                    "gflops": round(gflops, 2),
                }
            )

    # Verify replies across EVERY category!
    test_prompts = [
        "Hi",
        "How are you?",
        "Who are you?",
        "What is 2 + 2?",
        "What is 6 * 7?",
        "What is pi?",
        "Why is the sky blue?",
        "What is water made of?",
        "How many planets are in the solar system?",
        "What is the capital of Nepal?",
        "What is Mount Everest?",
        "What is the capital of Japan?",
        "What is Python?",
        "Write a fibonacci function",
        "What is Glassbox-100M?",
        "What is GQA?",
        "What is a KV cache?",
        "Tell me a joke",
        "Write a haiku",
        "Give me a fun fact",
    ]
    print("\n=== Live KV-Cached Basics Verification (20 Prompts) ===")
    for p in test_prompts:
        ans = decode_reply(model, tok, p)
        print(f"  Q: {p}\n  A: {ans}\n")

    # Save .gbx binary checkpoint
    gbx_path = ROOT / "checkpoints" / "basics_gqa_v1.gbx"
    gbx_path.parent.mkdir(parents=True, exist_ok=True)
    save_gbx_checkpoint(gbx_path, model, step=total_steps, tokens_seen=tokens_seen)
    print(f"Saved binary checkpoint: {gbx_path} ({gbx_path.stat().st_size / 1e6:.2f} MB)")

    # Export Int8 Base64 bundle for our new standalone Vercel chat app!
    bundle = {
        "format": "glassbox-web-v1",
        "model_name": "Glassbox-Basics 1.2M (6L GQA)",
        "dataset_id": "basics",
        "preset_id": "basics_gqa_1_2m",
        "step": total_steps,
        "tokens_seen": tokens_seen,
        "num_parameters": n_params,
        "final_loss": history[-1]["loss"],
        "final_val_loss": history[-1]["val_loss"],
        "topics_count": len(BASICS_QA),
        "qa_variants_count": len(all_examples),
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff,
            "context_len": cfg.context_len,
            "tie_weights": cfg.tie_weights,
            "rope_base": cfg.rope_base,
            "rmsnorm_eps": cfg.rmsnorm_eps,
        },
        "tokenizer": tok.to_dict(),
        "history": history,
        "weights_int8": pack_state_dict_int8(model),
    }
    out_path = ROOT / "chat-app" / "public" / "models" / "glassbox-basics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(bundle, separators=(",", ":"))
    out_path.write_text(raw, encoding="utf-8")
    print(f"Exported standalone web bundle: {out_path} ({len(raw) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
