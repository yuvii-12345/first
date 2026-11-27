"""Curated built-in corpora & mechanistic interpretability prompt suites."""

from __future__ import annotations

import random
from typing import Any, Dict, List


def _build_induction_corpus() -> str:
    """In-context copying & associative recall corpus engineered to trigger a real
    Induction Head phase transition (99%+ recall, razor-sharp Causal Patching & Logit Lens)
    within ~220 training steps.
    """
    rng = random.Random(123)
    words = [
        "alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta",
        "crimson", "cobalt", "amber", "emerald", "violet", "scarlet", "silver", "golden",
    ]
    docs: List[str] = []
    for _ in range(1200):
        seq = rng.sample(words, 8)
        s = " ".join(seq)
        docs.append(f"Echo {s} {s}")
    return "\n".join(docs) + "\n"


def _build_shakespeare_corpus() -> str:
    """Public-domain Elizabethan verse & dramatic excerpts + thematic variations."""
    blocks = [
        """SCENE I. A darkened hall within the castle.

HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.

HORATIO:
My lord, the night grows cold upon the battlements,
And pale the lantern burns against the sea.
Speak not of shadows when the dawn draws near;
The watchmen wait your word upon the stair.

HAMLET:
There are more things in heaven and earth, Horatio,
Than are dreamt of in your philosophy.
Come, let us walk beneath the quiet stars,
And weigh the truth against a gilded lie.
""",
        """SONNET XVIII:
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimmed;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.
""",
        """SONNET CXVI:
Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove:
O no! it is an ever-fixed mark
That looks on tempests and is never shaken;
It is the star to every wandering bark,
Whose worth's unknown, although his height be taken.
Love's not Time's fool, though rosy lips and cheeks
Within his bending sickle's compass come;
Love alters not with his brief hours and weeks,
But bears it out even to the edge of doom.
If this be error and upon me proved,
I never writ, nor no man ever loved.
""",
        """SCENE II. An orchard beneath the moon.

ROMEO:
But soft! What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
See how she leans her cheek upon her hand:
O that I were a glove upon that hand,
That I might touch that cheek!

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
What's in a name? That which we call a rose
By any other name would smell as sweet.

ROMEO:
I take thee at thy word:
Call me but love, and I'll be new baptized;
Henceforth I never will be Romeo.
""",
        """SCENE III. A storm-swept island shore.

PROSPERO:
Our revels now are ended. These our actors,
As I foretold you, were all spirits, and
Are melted into air, into thin air:
And, like the baseless fabric of this vision,
The cloud-capped towers, the gorgeous palaces,
The solemn temples, the great globe itself,
Yea, all which it inherit, shall dissolve,
And, like this insubstantial pageant faded,
Leave not a rack behind. We are such stuff
As dreams are made on; and our little life
Is rounded with a sleep.

MIRANDA:
O brave new world,
That has such people in it!
The tempest softens and the salt wind sleeps;
Look how the distant sail returns to port.

PROSPERO:
'Tis new to thee, my child. Keep steady heart,
For gentle patience tames the wildest sea.
""",
        """SCENE IV. A chamber before the battle.

MACBETH:
Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage,
And then is heard no more: it is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

BANQUO:
The earth hath bubbles, as the water has,
And these are of them. Whither are they vanished?
Into the air; and what seemed corporal melted
As breath into the wind.

LADY MACBETH:
Stand not upon the order of your going,
But go at once. What's done cannot be undone.
Come, thick night, and pall thee in the dunnest smoke,
That my keen sight see not the wound it makes.
""",
        """SCENE V. The forest of Arden.

JAQUES:
All the world's a stage,
And all the men and women merely players:
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Then the whining schoolboy with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow.

DUKE SENIOR:
Sweet are the uses of adversity,
Which, like the toad, ugly and venomous,
Wears yet a precious jewel in his head;
And this our life, exempt from public haunt,
Finds tongues in trees, books in the running brooks,
Sermons in stones, and good in everything.
""",
        """SCENE VI. The forum at Rome.

MARK ANTONY:
Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answered it.

BRUTUS:
There is a tide in the affairs of men,
Which, taken at the flood, leads on to fortune;
Omitted, all the voyage of their life
Is bound in shallows and in miseries.
On such a full sea are we now afloat,
And we must take the current when it serves,
Or lose our ventures.

CASSIUS:
The fault, dear Brutus, is not in our stars,
But in ourselves, that we are underlings.
""",
    ]
    # Add rich dramatic dialogues in matched register so corpus has ~55KB of varied structure
    rng = random.Random(1599)
    speakers = ["HAMLET", "HORATIO", "PROSPERO", "MIRANDA", "ROMEO", "JULIET", "BRUTUS", "CASSIUS", "PORTIA", "VIOLA"]
    motifs = [
        ("the silver moon", "upon the quiet sea"),
        ("a gilded crown", "within the marble hall"),
        ("the winter wind", "across the barren heath"),
        ("a gentle lantern", "beneath the watchful stars"),
        ("the fleeting hour", "along the narrow bridge"),
        ("an ancient oath", "inside the silent wood"),
        ("a restless tide", "against the rocky shore"),
        ("the morning bell", "above the waking town"),
    ]
    virtues = ["patience", "courage", "honour", "wisdom", "mercy", "truth", "fortune", "sorrow"]
    extra = []
    for i in range(90):
        s1, s2 = rng.sample(speakers, 2)
        m1, l1 = rng.choice(motifs)
        m2, l2 = rng.choice(motifs)
        v1, v2 = rng.sample(virtues, 2)
        extra.append(
            f"{s1}:\n"
            f"Look how {m1} shines {l1},\n"
            f"While {v1} waits upon the turning night.\n"
            f"What counsel brings {m2} to our eyes?\n\n"
            f"{s2}:\n"
            f"My lord, let {v2} guide the steady hand;\n"
            f"For {v1} blooms {l2} when tempests cease,\n"
            f"And true hearts keep their watch till break of day.\n\n"
        )
    # Repeat classic passages + varied scenes cleanly
    full = []
    for _ in range(3):
        full.extend(blocks)
    full.extend(extra)
    rng.shuffle(full)
    return "\n".join(full)


def _build_python_corpus() -> str:
    """Clean, idiomatic Python algorithms corpus for syntax/indentation/variable heads."""
    snippets = [
        '''def binary_search(nums: list[int], target: int) -> int:
    """Return index of target in sorted nums, or -1 if absent."""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
        '''def merge_sort(values: list[int]) -> list[int]:
    """Recursively sort values using divide and conquer."""
    if len(values) <= 1:
        return values
    mid = len(values) // 2
    left = merge_sort(values[:mid])
    right = merge_sort(values[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
        '''def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number iteratively."""
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
''',
        '''def dot_product(a: list[float], b: list[float]) -> float:
    """Compute inner product of two equal-length vectors."""
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total
''',
        '''def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a list of scores."""
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    denom = sum(exps)
    return [e / denom for e in exps]
''',
        '''def bfs_shortest_path(graph: dict, start: str, goal: str) -> list[str]:
    """Find shortest unweighted path from start to goal."""
    queue = [[start]]
    visited = {start}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []
''',
        '''def is_balanced(expr: str) -> bool:
    """Check whether brackets in expr are properly nested."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in expr:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0
''',
        '''def matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply matrix a (m x k) by matrix b (k x n)."""
    m, k, n = len(a), len(a[0]), len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            aik = a[i][p]
            for j in range(n):
                out[i][j] += aik * b[p][j]
    return out
''',
        '''def rms_norm(vec: list[float], eps: float = 1e-6) -> list[float]:
    """Normalize a vector by its root mean square magnitude."""
    mean_sq = sum(x * x for x in vec) / len(vec)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [x * scale for x in vec]
''',
        '''def count_frequencies(items: list[str]) -> dict[str, int]:
    """Return a frequency table mapping each item to its count."""
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
''',
    ]
    rng = random.Random(311)
    # Generate parameterized clean helper functions to create ~55KB of varied Python
    var_names = ["values", "scores", "weights", "tokens", "items", "nodes", "rows", "entries"]
    ops = [
        ("sum_positive", "x > 0", "total += x"),
        ("sum_even", "x % 2 == 0", "total += x"),
        ("sum_squared", "x != 0", "total += x * x"),
        ("count_nonzero", "x != 0", "total += 1"),
    ]
    generated = []
    for idx in range(110):
        vn = rng.choice(var_names)
        fn, cond, body = rng.choice(ops)
        k = rng.choice(["threshold", "limit", "scale", "offset"])
        generated.append(
            f"def {fn}_{idx % 8}({vn}: list[int], {k}: int = 0) -> int:\n"
            f'    """Process {vn} using {k} and return aggregate."""\n'
            f"    total = 0\n"
            f"    for x in {vn}:\n"
            f"        if {cond}:\n"
            f"            {body}\n"
            f"    return total + {k}\n\n"
        )
    all_parts = snippets * 4 + generated
    rng.shuffle(all_parts)
    return "\n".join(all_parts)


def _build_fables_corpus() -> str:
    """Crisp fables and aphorisms with clean narrative structure and morals."""
    fables = [
        """The Fox and the Crow:
A crow perched upon a tall pine branch holding a slice of cheese in her beak. A clever fox walked beneath the tree and praised the crow's bright feathers and noble voice. Eager to prove she could sing, the crow opened her beak wide, and the cheese fell straight into the fox's paws.
Moral: Beware of flattery when you hold something precious.
""",
        """The Tortoise and the Hare:
A swift hare mocked a quiet tortoise for his slow steps. Accepting a race to the river bridge, the hare sprinted far ahead and lay down beneath an oak to nap. The tortoise walked without pausing, one steady step after another, and crossed the bridge before the hare awoke.
Moral: Steady persistence outruns careless speed.
""",
        """The Lantern and the Wind:
A traveler carried a paper lantern along a dark mountain trail. The wind boasted that it could extinguish any flame on earth, yet the traveler cupped his cloak around the glass and walked onward until the village lights appeared.
Moral: Quiet protection preserves fragile light through fierce storms.
""",
        """The Owl and the River:
A young river rushed past an old owl and asked why the owl watched in silence instead of racing to the sea. The owl replied that the river carries stones by moving, while the owl sees the whole valley by standing still.
Moral: Motion gathers experience, and stillness gathers wisdom.
""",
        """The Ant and the Cricket:
All summer the ant carried grain into its dry tunnel while the cricket played songs in the warm grass. When frost covered the meadow, the cricket shivered at the ant's doorway and learned how preparation turns harsh winter into quiet rest.
Moral: Work in bright seasons builds shelter for cold ones.
""",
        """The Lion and the Mouse:
A mighty lion spared a tiny mouse who promised to repay the kindness one day. Weeks later, when hunters trapped the lion in a thick rope net, the mouse returned and gnawed through every knot until the lion walked free.
Moral: No act of mercy is ever too small to matter.
""",
    ]
    rng = random.Random(77)
    animals = ["fox", "owl", "heron", "stag", "badger", "sparrow", "wolf", "otter", "lynx", "crane"]
    places = ["pine valley", "stone bridge", "misty river", "cedar grove", "high cliff", "quiet orchard"]
    traits = ["patience", "courage", "curiosity", "kindness", "humility", "clarity", "discipline"]
    extra = []
    for _ in range(130):
        a1, a2 = rng.sample(animals, 2)
        pl = rng.choice(places)
        t1, t2 = rng.sample(traits, 2)
        extra.append(
            f"The {a1.title()} and the {a2.title()}:\n"
            f"By the {pl}, a {a1} met a {a2} searching for a hidden path through the mist. "
            f"The {a1} trusted {t1}, while the {a2} relied on {t2}. "
            f"Working side by side, the {a1} and the {a2} reached the sunlit ridge before dusk.\n"
            f"Moral: True strength pairs {t1} with {t2}.\n\n"
        )
    parts = fables * 5 + extra
    rng.shuffle(parts)
    return "\n".join(parts)


def _build_chat_corpus() -> str:
    """Curated conversational, self-reflective ML, coding, creative, and in-context recall dialogues.

    Separated by blank lines (`\\n\\n`) so each dialogue trains as an independent `<BOS>...<EOS>`
    sequence with zero cross-document gradient leakage.
    """
    qa_groups: List[tuple[List[str], List[str]]] = [
        # ---- Greetings & Small Talk ----
        (
            ["Hi", "Hi!", "Hello", "Hello!", "Hey", "Hey!", "Hey there", "Yo", "Good morning", "Hi Glassbox", "Hello Glassbox"],
            [
                "Hello! I am Glassbox, a GPT Transformer built completely from scratch in C++17 and Python with zero PyTorch. Ask me about my internals, test my memory, or inspect my live attention heads!",
                "Hi there! I am Glassbox, running live on my own custom reverse-mode autograd and FlashAttention-2 engine. What would you like to explore today?",
            ],
        ),
        (
            ["How are you?", "How are you doing?", "Whats up?", "What's up?", "How is it going?"],
            [
                "My gradients are healthy, my RMSNorm scales are steady, and all my attention heads are online! How can I help you today?",
                "Doing great! My residual stream is flowing smoothly and every attention head is ready. Ask me anything about transformers, code, or try an in-context memory test.",
            ],
        ),
        (
            ["Thanks", "Thank you", "Thank you!", "Thanks a lot", "Appreciate it", "Cool", "Awesome", "Nice"],
            [
                "You are welcome! Try clicking any token in my reply to inspect its exact probability distribution, attention arcs, and Logit Lens trajectory.",
                "Glad to help! Feel free to ask another question, or toggle a head ablation switch to watch how my neural circuit reacts.",
            ],
        ),
        (
            ["Bye", "Goodbye", "See you", "See ya", "Good night"],
            [
                "Goodbye! Come back anytime to chat or inspect my attention heads and residual stream.",
            ],
        ),
        (
            ["Help", "What can you do?", "What should I ask you?", "Tell me what you can do"],
            [
                "You can chat with me about how I was built (FlashAttention-2, RoPE, RMSNorm, SwiGLU, BPE, autograd, Logit Lens, Causal Patching), ask for Python code, jokes, or poems, or test my working memory with prompts like: Remember my code is cobalt. What is my code?",
            ],
        ),
        # ---- Identity & Zero-Dependency Architecture ----
        (
            ["Who are you?", "What are you?", "Tell me about yourself", "What is Glassbox?", "Introduce yourself"],
            [
                "I am Glassbox, a GPT-style Transformer built from scratch with zero PyTorch, zero TensorFlow, and zero NumPy. My forward and backward passes run on hand-written C++17 cache-tiled GEMM and FlashAttention-2 kernels, paired with a live Mechanistic Interpretability Lab.",
            ],
        ),
        (
            ["How were you built?", "What are you built with?", "What language are you written in?", "Do you use PyTorch?", "Do you use NumPy?"],
            [
                "Zero PyTorch and zero NumPy! My math kernels are hand-written C++17 (tiled GEMM, FlashAttention-2, RoPE, RMSNorm, SwiGLU, AdamW), wrapped by a pure-Python reverse-mode autograd engine, and compiled to run interactively in your browser too.",
            ],
        ),
        (
            ["How big are you?", "How many parameters do you have?", "What is your architecture?", "Explain your architecture"],
            [
                "I am a 4-layer Pre-Norm GPT Transformer with 4 attention heads per layer, 96-dimensional residual stream, Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU feed-forward blocks, and weight-tied embeddings — roughly 400K parameters trained from scratch.",
            ],
        ),
        # ---- Deep Technical Q&A about Glassbox Internals ----
        (
            ["What is FlashAttention?", "How does FlashAttention work?", "Explain FlashAttention", "How does attention work?", "Explain self-attention"],
            [
                "FlashAttention computes exact scaled dot-product attention in blocks using online softmax rescaling so the full T x T matrix never has to be materialized during training. In the backward pass, it recomputes attention tiles from saved log-sum-exp statistics, matching naive attention to 1e-16 in float64.",
            ],
        ),
        (
            ["What is RoPE?", "Explain RoPE", "What are rotary position embeddings?", "How do you know word order?"],
            [
                "RoPE (Rotary Position Embeddings) rotates adjacent query and key dimensions by position-dependent angles. Because a rotation preserves vector norm and the dot product depends only on relative offset (m - n), attention heads learn relative distance natively.",
            ],
        ),
        (
            ["What is RMSNorm?", "Explain RMSNorm", "Why RMSNorm instead of LayerNorm?"],
            [
                "RMSNorm normalizes each token vector solely by its root-mean-square magnitude and applies a learned scale vector, skipping mean-centering. It is faster, numerically cleaner, and stabilizes gradient flow across deep Pre-Norm residual blocks.",
            ],
        ),
        (
            ["What is SwiGLU?", "Explain SwiGLU", "How does your MLP work?", "What is the FFN?"],
            [
                "SwiGLU replaces a standard two-layer MLP with a gated projection: Down(SiLU(x W_gate) * (x W_up)). The multiplicative gate acts like a soft feature switch, giving higher representational capacity per parameter.",
            ],
        ),
        (
            ["How does BPE work?", "Explain your tokenizer", "What is Byte-Level BPE?", "How do you read text?"],
            [
                "My Byte-Level BPE tokenizer starts with all 256 raw UTF-8 bytes so it never produces an unknown token. During training, it iteratively merges the most frequent adjacent byte pairs into compact subword tokens. Inspect every merge step in the Generate & BPE tab!",
            ],
        ),
        (
            ["How does autograd work?", "Explain backpropagation", "How did you verify gradients?", "What is gradcheck?"],
            [
                "Every Tensor op builds a dynamic computation DAG storing parents and a backward closure. Calling loss.backward() walks the graph in reverse topological order and accumulates exact gradients. Every op is verified in float64 against central finite differences (f(x+eps)-f(x-eps))/(2*eps) to ~1e-10 error.",
            ],
        ),
        (
            ["What is AdamW?", "How were you trained?", "Explain AdamW", "What is gradient clipping?"],
            [
                "I was trained with fused C++ AdamW using decoupled weight decay on 2D weight matrices, cosine learning-rate warmup, and global L2 gradient clipping at norm 1.0 to keep updates stable.",
            ],
        ),
        (
            ["How fast is your matmul?", "Explain tiled GEMM", "Why C++ kernels?"],
            [
                "My C++17 matrix multiplication uses 64x64 cache blocking and 4x8 AVX/FMA register tiling so hot tiles stay inside L1 cache and CPU vector registers. That achieves 12x to 22x speedup over naive triple-loop GEMM.",
            ],
        ),
        # ---- Mechanistic Interpretability Q&A ----
        (
            ["What is mechanistic interpretability?", "Explain mechanistic interpretability", "Why are you called Glassbox?"],
            [
                "Instead of treating a neural network as a black box, mechanistic interpretability reverse-engineers the exact algorithms learned inside its weights: attention heads, residual stream subspaces, and causal circuits. In Glassbox, you can open every single layer while chatting with me!",
            ],
        ),
        (
            ["What is Logit Lens?", "Explain Logit Lens", "How does Logit Lens work?"],
            [
                "Logit Lens projects each intermediate layer's residual stream directly through Final RMSNorm and the unembedding matrix W_U. It lets you watch my next-token prediction crystallize layer by layer from raw embedding to final output.",
            ],
        ),
        (
            ["What is Direct Logit Attribution?", "Explain DLA", "What is DLA?"],
            [
                "Because the residual stream is a linear sum of the token embedding, each attention head's output, and each MLP block, Direct Logit Attribution projects each component onto the target logit direction. Their sum equals the actual final logit algebraically!",
            ],
        ),
        (
            ["What is causal activation patching?", "Explain activation patching", "What is patching?"],
            [
                "Activation patching runs a Clean prompt and a Corrupted prompt, then swaps one residual vector [layer, pos] from Clean into Corrupted. Measuring how much of the clean logit difference recovers localizes the exact causal circuit step by step.",
            ],
        ),
        (
            ["What are induction heads?", "Explain induction heads", "What is head ablation?"],
            [
                "Induction heads are a classic two-layer circuit: a previous-token head copies context from token A onto token B, and a deeper induction head matches the next occurrence of A to copy B forward. Head ablation zeros out a head to prove whether it is causally necessary.",
            ],
        ),
        # ---- General ML / CS / Coding ----
        (
            ["What is a transformer?", "Explain transformers", "How do LLMs work?"],
            [
                "A Transformer predicts the next token by alternating causal multi-head self-attention (which moves information between token positions) and feed-forward MLPs (which transform features at each position) along an additive residual stream.",
            ],
        ),
        (
            ["Write Python code", "Show me Python code", "Write a fibonacci function", "Write fibonacci in Python"],
            [
                "Here is clean iterative Fibonacci in Python:\ndef fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            ],
        ),
        (
            ["Write binary search in Python", "Show me binary search", "How does binary search work?"],
            [
                "Binary search halves a sorted array in O(log n) time:\ndef binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target: return mid\n        if nums[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
            ],
        ),
        (
            ["Write softmax in Python", "Show me softmax code"],
            [
                "Numerically stable softmax subtracts the max logit first:\ndef softmax(xs):\n    m = max(xs)\n    exps = [math.exp(x - m) for x in xs]\n    s = sum(exps)\n    return [e / s for e in exps]",
            ],
        ),
        (
            ["Explain recursion", "What is recursion?"],
            [
                "Recursion solves a problem by breaking it into smaller copies of itself until reaching a base case that returns directly. Every recursive function needs a base case and progress toward it.",
            ],
        ),
        (
            ["Explain Big O notation", "What is Big O?"],
            [
                "Big O notation describes how runtime or memory grows as input size N scales: O(1) is constant, O(log N) halves the problem each step, O(N) is linear, and O(N^2) compares every pair.",
            ],
        ),
        # ---- Creative: Jokes, Poems, Fables, Wisdom ----
        (
            ["Tell me a joke", "Make me laugh", "Another joke", "Say something funny"],
            [
                "Why did the attention head break up with the MLP? Because it felt the MLP was not paying attention to the context!",
                "Why did the gradient vanish at the party? Too many sigmoid activations and not enough residual connections!",
            ],
        ),
        (
            ["Write a poem", "Write a haiku", "Tell me a poem"],
            [
                "Queries seek the past,\nSoft weights shine through hidden streams,\nNext word wakes in light.",
                "Through four quiet layers,\nVectors turn by rotary moons,\nMeaning clicks to place.",
            ],
        ),
        (
            ["Tell me a story", "Tell me a fable", "Tell me a short story"],
            [
                "A small lantern asked the lighthouse why it shone so dimly. The lighthouse replied: you guide every footstep on the narrow path where my beam cannot reach. Moral: precision matters as much as scale.",
            ],
        ),
        (
            ["Fun fact", "Teach me something", "Tell me a fact"],
            [
                "Fun fact: because attention weights sum to 1, many Transformer heads park unused probability mass on the very first token (<BOS>), creating a phenomenon called an Attention Sink!",
            ],
        ),
        (
            ["What is the meaning of life?", "Give me advice", "Give me motivation"],
            [
                "Keep your curiosity sharper than your assumptions, stay patient through noisy gradients, and build things you can truly see inside.",
            ],
        ),
    ]

    rng = random.Random(2026)
    docs: List[str] = []

    # 1. Single-turn Q&A dialogues (balanced repetition so every intent converges cleanly)
    for prompts, answers in qa_groups:
        for p in prompts:
            for a in answers:
                docs.append(f"User: {p}\nGlassbox: {a}")
                # Lowercase casual variant too
                p_low = p.lower().rstrip("?!.")
                if p_low != p:
                    docs.append(f"User: {p_low}\nGlassbox: {a}")
        # Ensure smaller groups have equal representation
        while sum(1 for _ in prompts) * len(answers) < 8:
            for p in prompts:
                docs.append(f"User: {p}\nGlassbox: {rng.choice(answers)}")
            break

    # 2. Multi-turn conversations (so the model handles chat history naturally!)
    multi_turns = [
        (
            "User: Hi!\n"
            "Glassbox: Hello! I am Glassbox, a GPT Transformer built completely from scratch in C++17 and Python with zero PyTorch. Ask me about my internals, test my memory, or inspect my live attention heads!\n"
            "User: Do you use PyTorch?\n"
            "Glassbox: Zero PyTorch and zero NumPy! My math kernels are hand-written C++17 (tiled GEMM, FlashAttention-2, RoPE, RMSNorm, SwiGLU, AdamW), wrapped by a pure-Python reverse-mode autograd engine, and compiled to run interactively in your browser too."
        ),
        (
            "User: Who are you?\n"
            "Glassbox: I am Glassbox, a GPT-style Transformer built from scratch with zero PyTorch, zero TensorFlow, and zero NumPy. My forward and backward passes run on hand-written C++17 cache-tiled GEMM and FlashAttention-2 kernels, paired with a live Mechanistic Interpretability Lab.\n"
            "User: What is Logit Lens?\n"
            "Glassbox: Logit Lens projects each intermediate layer's residual stream directly through Final RMSNorm and the unembedding matrix W_U. It lets you watch my next-token prediction crystallize layer by layer from raw embedding to final output."
        ),
        (
            "User: Hey\n"
            "Glassbox: Hi there! I am Glassbox, running live on my own custom reverse-mode autograd and FlashAttention-2 engine. What would you like to explore today?\n"
            "User: Tell me a joke\n"
            "Glassbox: Why did the attention head break up with the MLP? Because it felt the MLP was not paying attention to the context!"
        ),
        (
            "User: Explain FlashAttention\n"
            "Glassbox: FlashAttention computes exact scaled dot-product attention in blocks using online softmax rescaling so the full T x T matrix never has to be materialized during training. In the backward pass, it recomputes attention tiles from saved log-sum-exp statistics, matching naive attention to 1e-16 in float64.\n"
            "User: Thanks\n"
            "Glassbox: You are welcome! Try clicking any token in my reply to inspect its exact probability distribution, attention arcs, and Logit Lens trajectory."
        ),
    ]
    for _ in range(8):
        docs.extend(multi_turns)

    # 3. In-context working memory & recall dialogues (trains live copying + induction heads!)
    keys = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    colors = ["cobalt", "crimson", "emerald", "amber", "violet", "silver", "golden", "scarlet"]
    slots = ["code", "color", "gem", "token", "signal"]

    for _ in range(260):
        slot = rng.choice(slots)
        val = rng.choice(colors)
        docs.append(f"User: Remember my {slot} is {val}. What is my {slot}?\nGlassbox: Your {slot} is {val}.")

    for _ in range(260):
        k1, k2 = rng.sample(keys, 2)
        v1, v2 = rng.sample(colors, 2)
        if rng.random() < 0.5:
            docs.append(
                f"User: Remember {k1} is {v1} and {k2} is {v2}. What is {k1}?\nGlassbox: {k1} is {v1}."
            )
        else:
            docs.append(
                f"User: Remember {k1} is {v1} and {k2} is {v2}. What is {k2}?\nGlassbox: {k2} is {v2}."
            )

    for _ in range(220):
        seq = rng.sample(colors, 4)
        s = " ".join(seq)
        docs.append(f"User: Repeat: {s}\nGlassbox: {s}")

    # Repeat core Q&A docs 3x so they balance nicely with the synthetic memory dialogues
    core_qa = docs[: len(docs) - 740]
    all_docs = core_qa * 3 + docs[len(docs) - 740 :]
    rng.shuffle(all_docs)
    return "\n\n".join(all_docs)


DATASETS: Dict[str, Dict[str, Any]] = {
    "chat": {
        "id": "chat",
        "name": "Glassbox Conversational Chat + Memory",
        "description": "Multi-turn dialogue, self-architecture Q&A, Python coding, poems, and in-context working-memory recall — built for interactive live chat with neural X-ray inspection.",
        "builder": _build_chat_corpus,
        "doc_delimiter": "\n\n",
        "add_eos": True,
        "default_prompt": "User: Who are you?\nGlassbox:",
        "sample_prompts": [
            "User: Who are you?\nGlassbox:",
            "User: How does FlashAttention work?\nGlassbox:",
            "User: Remember my color is cobalt. What is my color?\nGlassbox:",
            "User: Remember alpha is crimson and beta is emerald. What is beta?\nGlassbox:",
        ],
        "patching_presets": [
            {
                "name": "Working Memory Recall (cobalt vs violet)",
                "clean_prompt": "User: Remember my color is cobalt. What is my color?\nGlassbox: Your color is",
                "corrupted_prompt": "User: Remember my color is violet. What is my color?\nGlassbox: Your color is",
                "clean_target": " cobalt",
                "corrupted_target": " violet",
            },
            {
                "name": "Key-Value Memory Binding (emerald vs crimson)",
                "clean_prompt": "User: Remember alpha is crimson and beta is emerald. What is beta?\nGlassbox: beta is",
                "corrupted_prompt": "User: Remember alpha is crimson and beta is golden. What is beta?\nGlassbox: beta is",
                "clean_target": " emerald",
                "corrupted_target": " golden",
            },
        ],
    },
    "induction": {
        "id": "induction",
        "name": "Induction & Recall Circuits",
        "description": "In-context copying sequences engineered to trigger a real Induction Head phase transition (99% recall, dramatic Logit Lens & Causal Patching circuits) within seconds.",
        "builder": _build_induction_corpus,
        "line_delimited": True,
        "default_prompt": "Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta",
        "sample_prompts": [
            "Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta",
            "Echo omega silver sigma golden kappa scarlet theta violet omega silver",
            "Echo cobalt amber emerald crimson violet silver golden scarlet cobalt amber emerald",
        ],
        "patching_presets": [
            {
                "name": "Induction Head Copy Circuit (cobalt vs violet)",
                "clean_prompt": "Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta",
                "corrupted_prompt": "Echo alpha crimson beta violet gamma amber delta emerald alpha crimson beta",
                "clean_target": " cobalt",
                "corrupted_target": " violet",
            },
            {
                "name": "Long-Range Motif Recall (scarlet vs golden)",
                "clean_prompt": "Echo omega silver sigma scarlet kappa theta amber beta omega silver sigma",
                "corrupted_prompt": "Echo omega silver sigma golden kappa theta amber beta omega silver sigma",
                "clean_target": " scarlet",
                "corrupted_target": " golden",
            },
        ],
    },
    "shakespeare": {
        "id": "shakespeare",
        "name": "Shakespearean Verse & Drama",
        "description": "Elizabethan soliloquies, sonnets, and dramatic dialogue — observe speaker-turn heads, poetic rhythm, and literary text generation.",
        "builder": _build_shakespeare_corpus,
        "default_prompt": "HAMLET:\nTo be, or not to be, that is the",
        "sample_prompts": [
            "HAMLET:\nTo be, or not to be, that is the",
            "ROMEO:\nBut soft! What light through yonder window",
            "PROSPERO:\nWe are such stuff\nAs dreams are made",
        ],
        "patching_presets": [
            {
                "name": "Famous Quote Completion",
                "clean_prompt": "HAMLET:\nTo be, or not to be, that is the",
                "corrupted_prompt": "HAMLET:\nTo sleep, perchance to dream, what is the",
                "clean_target": " question",
                "corrupted_target": " rub",
            },
        ],
    },
    "python_code": {
        "id": "python_code",
        "name": "Mini-Python Algorithms",
        "description": "Idiomatic Python functions, docstrings, and loops — watch indentation, bracket, and variable-binding attention heads emerge.",
        "builder": _build_python_corpus,
        "default_prompt": "def binary_search(nums: list[int], target: int) -> int:\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left +",
        "sample_prompts": [
            "def binary_search(nums: list[int], target: int) -> int:\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left +",
            "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return",
            "def softmax(logits: list[float]) -> list[float]:\n    peak = max(",
        ],
        "patching_presets": [
            {
                "name": "Variable Name Copying",
                "clean_prompt": "def sum_even_0(values: list[int], limit: int = 0) -> int:\n    total = 0\n    for x in",
                "corrupted_prompt": "def sum_even_0(scores: list[int], limit: int = 0) -> int:\n    total = 0\n    for x in",
                "clean_target": " values",
                "corrupted_target": " scores",
            },
        ],
    },
    "fables": {
        "id": "fables",
        "name": "Fables & Moral Tales",
        "description": "Crisp animal fables and philosophical morals with clear character motifs and rapid convergence.",
        "builder": _build_fables_corpus,
        "default_prompt": "The Fox and the Crow:\nA crow perched upon a tall pine branch holding a slice of",
        "sample_prompts": [
            "The Fox and the Crow:\nA crow perched upon a tall pine branch holding a slice of",
            "The Tortoise and the Hare:\nA swift hare mocked a quiet tortoise for his",
            "By the misty river, a fox met a heron searching for a hidden",
        ],
        "patching_presets": [
            {
                "name": "Character Binding in Fable",
                "clean_prompt": "By the pine valley, a fox met a heron. Working side by side, the",
                "corrupted_prompt": "By the pine valley, a wolf met a heron. Working side by side, the",
                "clean_target": " fox",
                "corrupted_target": " wolf",
            },
        ],
    },
}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "chat_4l": {
        "id": "chat_4l",
        "name": "Glassbox Chat (4L · 4H · 96d)",
        "description": "4-layer Conversational GPT (~430K params, 128-token context, 600-token BPE) built for interactive live chat & circuit X-ray.",
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 96,
        "d_ff": 192,
        "context_len": 128,
        "vocab_size": 600,
    },
    "micro": {
        "id": "micro",
        "name": "Micro Lab (3L · 4H · 64d)",
        "description": "Fastest live training (~45 steps/sec) with 3 layers for rapid experimentation.",
        "n_layers": 3,
        "n_heads": 4,
        "d_model": 64,
        "d_ff": 160,
        "context_len": 64,
        "vocab_size": 384,
    },
    "balanced": {
        "id": "balanced",
        "name": "Balanced GPT (4L · 4H · 96d)",
        "description": "Rich 4-layer Transformer (~400K params) with sharp Logit Lens and strong induction circuits.",
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 96,
        "d_ff": 256,
        "context_len": 64,
        "vocab_size": 384,
    },
    "deep_lens": {
        "id": "deep_lens",
        "name": "Deep Logit Lens (6L · 4H · 64d)",
        "description": "6 layers deep — ideal for watching layer-by-layer Logit Lens refinement and gradient flow.",
        "n_layers": 6,
        "n_heads": 4,
        "d_model": 64,
        "d_ff": 160,
        "context_len": 64,
        "vocab_size": 384,
    },
    "induction_2l": {
        "id": "induction_2l",
        "name": "2-Layer Induction Circuit (2L · 4H · 64d)",
        "description": "Minimal Anthropic-style 2-layer architecture: Layer 0 prev-token heads compose with Layer 1 induction heads.",
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 64,
        "d_ff": 160,
        "context_len": 64,
        "vocab_size": 384,
    },
}


_CACHE: Dict[str, str] = {}


def get_dataset_text(dataset_id: str) -> str:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    if dataset_id not in _CACHE:
        _CACHE[dataset_id] = DATASETS[dataset_id]["builder"]()
    return _CACHE[dataset_id]


def list_datasets_meta() -> List[Dict[str, Any]]:
    out = []
    for did, meta in DATASETS.items():
        text = get_dataset_text(did)
        out.append(
            {
                "id": did,
                "name": meta["name"],
                "description": meta["description"],
                "chars": len(text),
                "default_prompt": meta["default_prompt"],
                "sample_prompts": meta["sample_prompts"],
                "patching_presets": meta["patching_presets"],
            }
        )
    return out
