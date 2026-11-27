import type { QAPair } from "../types";

export interface PromptCategory {
  id: string;
  label: string;
  prompts: string[];
}

export const STARTER_CATEGORIES: PromptCategory[] = [
  {
    id: "everyday",
    label: "Everyday",
    prompts: ["Hi", "How are you?", "What can you do?", "Tell me a joke"],
  },
  {
    id: "math_sci",
    label: "Math & Science",
    prompts: ["What is 6 * 7?", "Why is the sky blue?", "What is DNA?", "How many planets are in the solar system?"],
  },
  {
    id: "world",
    label: "World & Nepal",
    prompts: ["What is the capital of Nepal?", "What is Mount Everest?", "What is the capital of Japan?", "Give me a fun fact"],
  },
  {
    id: "coding",
    label: "Coding & AI",
    prompts: ["Write a fibonacci function", "What is Python?", "What is Glassbox-100M?", "What is a KV cache?"],
  },
  {
    id: "creative",
    label: "Creative",
    prompts: ["Write a haiku", "Write a short story", "Give me a motivational quote", "Say something cool"],
  },
];

export const HERO_SUGGESTIONS = [
  { title: "Say hello & meet Glassbox", prompt: "Hi", badge: "Basics" },
  { title: "Ask world & Nepal trivia", prompt: "What is the capital of Nepal?", badge: "Geography" },
  { title: "Write clean Python code", prompt: "Write a fibonacci function", badge: "Coding" },
  { title: "Ask science fundamentals", prompt: "Why is the sky blue?", badge: "Science" },
  { title: "Explore Glassbox-100M", prompt: "What is Glassbox-100M?", badge: "100M LLM" },
  { title: "Tell a nerdy AI joke", prompt: "Tell me a joke", badge: "Fun" },
];

export interface LessonPack {
  id: string;
  name: string;
  description: string;
  pairs: Omit<QAPair, "id">[];
}

export const LESSON_PACKS: LessonPack[] = [
  {
    id: "personal",
    name: "Personalize Me",
    description: "Teach Glassbox your name, favorite hobby, and a custom greeting.",
    pairs: [
      {
        question: "What is my name?",
        answer: "Your name is Yuvraj — welcome back to Glassbox AI!",
      },
      {
        question: "What is my favorite project?",
        answer: "Your favorite project is building Glassbox, a pure from-scratch 100M LLM!",
      },
    ],
  },
  {
    id: "space",
    name: "Space & Universe",
    description: "Teach new astronomy facts about Mars, the Moon, and the Milky Way.",
    pairs: [
      {
        question: "How far away is the Moon?",
        answer: "The Moon orbits Earth at an average distance of about 384,400 kilometers (238,855 miles).",
      },
      {
        question: "Why is Mars red?",
        answer: "Mars looks red because its dusty surface is rich in iron oxide — essentially rust!",
      },
    ],
  },
  {
    id: "python_extra",
    name: "Python Snippets",
    description: "Teach extra handy Python coding patterns.",
    pairs: [
      {
        question: "How do I check if a number is even in Python?",
        answer: "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
      },
      {
        question: "How do I sort a list in Python?",
        answer: "```python\nnums = [5, 2, 9, 1]\nnums.sort()  # [1, 2, 5, 9]\n```",
      },
    ],
  },
  {
    id: "nepal_extra",
    name: "Nepal Basics",
    description: "Teach extra geography & culture facts about Nepal.",
    pairs: [
      {
        question: "What is the currency of Nepal?",
        answer: "The official currency of Nepal is the Nepalese Rupee (NPR).",
      },
      {
        question: "What is Pokhara famous for?",
        answer: "Pokhara is famous for Phewa Lake, breathtaking views of the Annapurna and Machhapuchhre peaks, and adventure trekking!",
      },
    ],
  },
];
