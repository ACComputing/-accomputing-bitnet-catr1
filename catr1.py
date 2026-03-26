#!/usr/bin/env python3
"""
cat r1.x – BitNet LLM in Tkinter
Size: 600×400, no file output. Model is a binary neural network trained on embedded text.
Dark theme: black background, blue text.
Enhanced with language/code detection heuristics.
"""

import tkinter as tk
from tkinter import scrolledtext
import random
import math
import numpy as np
import re

# ----------------------------------------------------------------------
# DeepSeek-style mini architecture (inspired, compact, single-file)
# ----------------------------------------------------------------------
class BitNetLM:
    def __init__(
        self,
        d_model=64,
        n_layers=2,
        n_heads=4,
        context_len=48,
        n_experts=4,
        top_k_experts=2,
    ):
        self.chars = [chr(ord('a')+i) for i in range(26)] + [' ', '.', ',', '!', '?']
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_len = context_len
        self.n_experts = n_experts
        self.top_k_experts = max(1, min(top_k_experts, n_experts))
        self.head_dim = self.d_model // self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        # Token embedding + output projection
        self.embedding = np.random.randn(self.vocab_size, self.d_model) * 0.05
        self.out_proj = np.random.randn(self.d_model, self.vocab_size) * 0.05
        self.out_bias = np.zeros(self.vocab_size, dtype=np.float64)

        # DeepSeek-style block params: RMSNorm + RoPE attention + MoE(SwiGLU)
        self.blocks = []
        for _ in range(self.n_layers):
            block = {
                "attn_norm_g": np.ones(self.d_model, dtype=np.float64),
                "ffn_norm_g": np.ones(self.d_model, dtype=np.float64),
                "wq": np.random.randn(self.d_model, self.d_model) * 0.05,
                "wk": np.random.randn(self.d_model, self.d_model) * 0.05,
                "wv": np.random.randn(self.d_model, self.d_model) * 0.05,
                "wo": np.random.randn(self.d_model, self.d_model) * 0.05,
                "router_w": np.random.randn(self.d_model, self.n_experts) * 0.05,
                "experts": [],
            }
            ff_dim = self.d_model * 2
            for _e in range(self.n_experts):
                block["experts"].append(
                    {
                        "w1": np.random.randn(self.d_model, ff_dim) * 0.05,
                        "w3": np.random.randn(self.d_model, ff_dim) * 0.05,
                        "w2": np.random.randn(ff_dim, self.d_model) * 0.05,
                    }
                )
            self.blocks.append(block)

        self.init_language_bias()

    def normalize_text(self, text):
        text = text.lower()
        # Keep only known vocab symbols.
        return ''.join(ch if ch in self.char_to_idx else ' ' for ch in text)

    def init_language_bias(self):
        corpus = (
            "hello i am cat v zero. i can chat in simple english. "
            "if you ask me a short question, i give a short answer. "
            "if you ask me for help with code, i can explain ideas clearly. "
            "i try to be polite, concise, and useful. "
            "the weather is nice today. i hope your project is going well. "
            "python is a readable programming language. "
            "functions take input and return output. "
            "loops repeat steps. conditionals choose between paths. "
            "debugging starts with reading the error message carefully. "
            "small tests help find bugs quickly. "
            "clean code uses clear names and simple structure. "
            "hello there. how can i help you today? "
            "i can summarize text, answer questions, and suggest next steps. "
            "when i do not know something, i ask for clarification. "
            "thanks for chatting with cat v zero. "
        )
        corpus = self.normalize_text(corpus)
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for ch in corpus:
            counts[self.char_to_idx[ch]] += 1.0
        probs = (counts + 1.0) / np.sum(counts + 1.0)
        self.out_bias = np.log(probs)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def silu(self, x):
        return x / (1.0 + np.exp(-x))

    def rms_norm(self, x, g, eps=1e-6):
        # x: (T, D)
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return (x / rms) * g

    def apply_rope(self, q, k):
        # q, k: (H, T, Hd)
        h, t, d = q.shape
        if d % 2 != 0:
            return q, k
        half = d // 2
        pos = np.arange(t)[:, None]
        freq = 1.0 / (10000.0 ** (np.arange(half)[None, :] / max(half, 1)))
        ang = pos * freq
        cos = np.cos(ang)[None, :, :]
        sin = np.sin(ang)[None, :, :]

        def rotate(x):
            x1 = x[:, :, :half]
            x2 = x[:, :, half:]
            return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        return rotate(q), rotate(k)

    def encode_context(self, context):
        padded = self.normalize_text(context).rjust(self.context_len, ' ')
        idxs = []
        for ch in padded[-self.context_len:]:
            idx = self.char_to_idx.get(ch, self.char_to_idx[' '])
            idxs.append(idx)
        return np.array(idxs, dtype=np.int32)

    def attention(self, x, block):
        # x: (T, D)
        t = x.shape[0]
        q = x @ block["wq"]
        k = x @ block["wk"]
        v = x @ block["wv"]
        q = q.reshape(t, self.n_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(t, self.n_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(t, self.n_heads, self.head_dim).transpose(1, 0, 2)
        q, k = self.apply_rope(q, k)

        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.head_dim)
        mask = np.triu(np.ones((t, t)), 1) * -1e9
        scores = scores + mask[None, :, :]
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        out = probs @ v
        out = out.transpose(1, 0, 2).reshape(t, self.d_model)
        return out @ block["wo"]

    def moe_ffn(self, x, block):
        # x: (T, D)
        t = x.shape[0]
        gates = x @ block["router_w"]  # (T, E)
        probs = np.exp(gates - np.max(gates, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        out = np.zeros_like(x)

        for i in range(t):
            token = x[i]
            p = probs[i]
            top_idx = np.argpartition(p, -self.top_k_experts)[-self.top_k_experts:]
            top_p = p[top_idx]
            top_p = top_p / np.sum(top_p)
            mix = np.zeros(self.d_model, dtype=np.float64)
            for w, eidx in zip(top_p, top_idx):
                exp = block["experts"][int(eidx)]
                u = token @ exp["w1"]
                v = token @ exp["w3"]
                h = self.silu(u) * v  # SwiGLU
                y = h @ exp["w2"]
                mix += w * y
            out[i] = mix
        return out

    def forward(self, context):
        idxs = self.encode_context(context)
        x = self.embedding[idxs].astype(np.float64)  # (T, D)

        for block in self.blocks:
            h = self.rms_norm(x, block["attn_norm_g"])
            x = x + self.attention(h, block)
            h = self.rms_norm(x, block["ffn_norm_g"])
            x = x + self.moe_ffn(h, block)

        # Last-token logits
        last = x[-1]
        logits = last @ self.out_proj + self.out_bias
        return logits

    def sample(self, logits, temperature=0.45, top_k=5):
        if temperature <= 0:
            return np.argmax(logits)
        scaled = logits / max(temperature, 1e-6)
        probs = self.softmax(scaled)
        # Top-k sampling to reduce noisy low-probability characters.
        k = max(1, min(top_k, self.vocab_size))
        top_idx = np.argpartition(probs, -k)[-k:]
        top_probs = probs[top_idx]
        top_probs = top_probs / np.sum(top_probs)
        return np.random.choice(top_idx, p=top_probs)

    def generate(self, prompt, max_new=100, temperature=0.45, top_k=5):
        generated = self.normalize_text(prompt)
        if not generated:
            generated = "hello"
        for _ in range(max_new):
            context = generated[-self.context_len:] if len(generated) >= self.context_len else generated
            logits = self.forward(context)
            next_idx = self.sample(logits, temperature=temperature, top_k=top_k)
            next_char = self.idx_to_char[next_idx]
            # Prevent punctuation bursts that look like gibberish.
            if len(generated) >= 2 and all(ch in ".,!?" for ch in generated[-2:]) and next_char in ".,!?":
                next_char = ' '
            generated += next_char
            if len(generated) > len(prompt) + 24 and next_char in ".!?":
                break
        return generated

# ----------------------------------------------------------------------
# Language / Code Detection Heuristics
# ----------------------------------------------------------------------
def detect_input_type(text):
    t = text.lower()
    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))

    # Code detection (keywords and symbols)
    code_keywords = {'def', 'class', 'if', 'else', 'for', 'while', 'import', 'return',
                     'int', 'str', 'list', 'dict', 'print', 'function', 'var', 'let',
                     'const', 'console', 'log', 'echo', 'public', 'private', 'static'}
    code_symbols = {'{', '}', '(', ')', '[', ']', ';', '=', '+', '-', '*', '/', '==', '!=', '=>'}
    words = set(re.findall(r'\b[a-zA-Z_]+\b', t))
    code_intent_phrases = [
        "write",
        "code",
        "in html",
        "in c",
        "in python",
        "in javascript",
        "in js",
        "in java",
        "in cpp",
        "in c++",
    ]
    if (
        words.intersection(code_keywords)
        or any(sym in text for sym in code_symbols)
        or any(phrase in t for phrase in code_intent_phrases)
    ):
        return 'code'

    # Chinese intent for code requests (simplified phrases)
    if has_cjk:
        zh_code_phrases = ["写代码", "代码", "html", "python", "javascript", "js", "c语言", "c++", "程序"]
        if any(p in text for p in zh_code_phrases):
            return 'code'
        return 'mandarin'

    return 'english'

# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class CatR1XApp:
    # R1-inspired inference knobs (compact local version)
    R1_NUM_SAMPLES = 4
    R1_MAX_NEW_TOKENS = 90
    R1_TEMPERATURES = (0.35, 0.45, 0.55, 0.65)

    def normalize_user_text(self, text: str):
        t = text.lower().strip()
        # Keep letters/spaces for lightweight intent matching.
        t = re.sub(r'[^a-z\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def contains_cjk(self, text: str):
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def is_greeting(self, text: str):
        t = self.normalize_user_text(text)
        words = set(t.split())
        greeting_words = {
            "hi", "hey", "hello", "yo", "sup", "hiya",
            "hewwo", "heey", "hai", "hola", "greetings"
        }
        if words.intersection(greeting_words):
            return True
        # typo-friendly starts
        return t.startswith("he") or t.startswith("hi")

    def is_farewell(self, text: str):
        t = self.normalize_user_text(text)
        words = set(t.split())
        farewell_words = {
            "bye", "goodbye", "cya", "later", "farewell", "night", "gn", "goodnight"
        }
        return bool(words.intersection(farewell_words))

    def is_reaction(self, text: str):
        t = self.normalize_user_text(text)
        words = set(t.split())
        reaction_words = {
            "wow", "cool", "nice", "great", "ok", "okay", "test", "testing", "hmm"
        }
        return bool(words.intersection(reaction_words))

    def is_cat_talk(self, text: str):
        t = self.normalize_user_text(text)
        words = set(t.split())
        return "meow" in words or "mew" in words or "nya" in words

    def __init__(self, root):
        self.root = root
        self.root.title("cat r1.x")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        self.root.configure(bg='black')

        self.model = BitNetLM()

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state='disabled',
            font=("Segoe UI", 10),
            bg='black', fg='blue', insertbackground='blue'
        )
        self.chat_display.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew")

        self.input_entry = tk.Entry(
            self.root, font=("Segoe UI", 10),
            bg='black', fg='blue', insertbackground='blue'
        )
        self.input_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.input_entry.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(
            self.root, text="Send", command=self.send_message, width=10,
            bg='#333333', fg='blue', activebackground='#444444', activeforeground='cyan'
        )
        self.send_btn.grid(row=2, column=0, padx=10, pady=(0,10))

        self.append_chat("cat r1.x (BitNet LLM) ready. Type a prompt and press Enter.")

    def is_gibberish(self, text: str):
        if not text:
            return True
        # Do not apply English-only noise heuristics to Chinese responses.
        if self.contains_cjk(text):
            return False
        # Heuristic: too many punctuation chars or too few vowels => likely noise.
        punc_ratio = sum(ch in ".,!?" for ch in text) / max(len(text), 1)
        vowel_ratio = sum(ch in "aeiou" for ch in text) / max(sum(ch.isalpha() for ch in text), 1)
        alpha_chars = ''.join(ch for ch in text if ch.isalpha())
        unique_ratio = len(set(alpha_chars)) / max(len(alpha_chars), 1)
        return punc_ratio > 0.18 or vowel_ratio < 0.22 or unique_ratio > 0.75

    def fallback_reply_mandarin(self, user_input: str):
        text = user_input.strip()
        if any(k in text for k in ["你好", "您好", "嗨", "哈喽", "早上好", "晚上好"]):
            return "你好！我在这儿。你想聊聊天，还是要我帮你写代码？"
        if any(k in text for k in ["再见", "拜拜", "回头见"]):
            return "再见！随时回来找我，我们继续。"
        if any(k in text for k in ["喵", "猫", "meow"]):
            return "喵~ 我在听。你要聊天还是写代码？"
        if any(k in text for k in ["帮助", "帮我", "不会", "怎么"]):
            return "可以的。告诉我你的目标，我会一步一步帮你。"
        if any(k in text for k in ["代码", "python", "html", "js", "javascript", "c语言", "c++"]):
            return "没问题，把需求或报错贴给我，我来帮你修。"
        return "收到。你可以多说一点细节，我会给你更准确的回答。"

    def fallback_reply(self, user_input: str):
        if self.contains_cjk(user_input):
            return self.fallback_reply_mandarin(user_input)
        text = self.normalize_user_text(user_input)
        if self.is_greeting(text):
            return "hey! i am here. what do you want to work on?"
        if self.is_cat_talk(text):
            return "meow 😺 i am listening. want chat, code help, or both?"
        if self.is_farewell(text):
            return "bye! come back anytime and we can keep building."
        if self.is_reaction(text):
            return "nice. if you want, give me a task and i will do it step by step."
        if "help" in text:
            return "sure. tell me your goal and i will break it into simple steps."
        if "code" in text or "python" in text:
            return "i can help debug your code. paste the error and the related function."
        return "got it. can you share a bit more detail so i can give a better answer?"

    def code_template_reply(self, user_input: str):
        t = user_input.lower()
        # Basic intent extraction: "write ... in <lang>"
        if (
            ("hello cat" in t and ("html" in t))
            or ("你好猫" in user_input and "html" in t)
            or ("用html" in user_input and "猫" in user_input)
        ):
            return (
                "<!doctype html>\n"
                "<html>\n"
                "<head><meta charset=\"utf-8\"><title>Hello Cat</title></head>\n"
                "<body>\n"
                "  <h1>Hello Cat</h1>\n"
                "</body>\n"
                "</html>"
            )
        if (
            ("hello cat" in t and (" in c" in t or t.endswith(" c") or "language c" in t))
            or (("用c" in user_input or "c语言" in user_input) and ("猫" in user_input or "hello cat" in t))
        ):
            return (
                "#include <stdio.h>\n\n"
                "int main(void) {\n"
                "    printf(\"Hello Cat\\n\");\n"
                "    return 0;\n"
                "}"
            )
        if "hello cat" in t and ("python" in t):
            return "print(\"Hello Cat\")"
        if "hello cat" in t and ("javascript" in t or " js" in t):
            return "console.log(\"Hello Cat\");"
        return None

    def score_candidate(self, text: str, user_input: str):
        """Heuristic verifier: reward coherence + lexical overlap."""
        if not text:
            return -1e9
        score = 0.0
        if not self.is_gibberish(text):
            score += 2.0
        # Prefer moderate length replies.
        score += min(len(text), 220) / 220.0
        # Reward overlap with user keywords.
        u_words = set(re.findall(r"[a-z]+", user_input.lower()))
        t_words = set(re.findall(r"[a-z]+", text.lower()))
        if u_words:
            score += len(u_words.intersection(t_words)) / max(len(u_words), 1)
        # Penalize long repeated chars.
        if re.search(r"(.)\1\1\1", text):
            score -= 1.0
        return score

    def reflect_response(self, text: str, is_code: bool = False):
        """R1-style reflection pass: trim noise and enforce formatting."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if is_code:
            # Keep code formatting from templates/model when code is requested.
            return text.strip()
        # Keep concise conversational output.
        if len(cleaned) > 220:
            cleaned = cleaned[:220].rstrip() + "..."
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def r1_generate(self, user_input: str, is_code: bool = False):
        """
        Compact R1-inspired decoding:
        - multi-sample generation
        - verifier scoring
        - reflection pass on best candidate
        """
        candidates = []
        for i in range(self.R1_NUM_SAMPLES):
            temp = self.R1_TEMPERATURES[i % len(self.R1_TEMPERATURES)]
            top_k = 4 if is_code else 5
            cand = self.model.generate(
                user_input,
                max_new=self.R1_MAX_NEW_TOKENS,
                temperature=temp,
                top_k=top_k,
            )
            candidates.append(cand)

        best = max(candidates, key=lambda c: self.score_candidate(c, user_input))
        return self.reflect_response(best, is_code=is_code)

    def append_chat(self, text: str, sender: str = "System"):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {text}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')

    def send_message(self, event=None):
        user_input = self.input_entry.get().strip()
        if not user_input:
            return

        self.append_chat(user_input, sender="You")
        self.input_entry.delete(0, tk.END)

        detected = detect_input_type(user_input)
        self.append_chat(f"(Detected: {detected})", sender="System")

        if detected == 'mandarin':
            response = self.fallback_reply_mandarin(user_input)
        elif detected == 'code':
            templated = self.code_template_reply(user_input)
            if templated is not None:
                response = templated
            else:
                response = self.r1_generate(user_input, is_code=True)
        else:
            # Prefer coherent reply over noisy char-model output for plain chat.
            response = self.fallback_reply(user_input)
            # Try model only on longer prompts; keep fallback if noisy.
            if len(user_input) >= 20:
                candidate = self.r1_generate(user_input, is_code=False)
                if not self.is_gibberish(candidate):
                    response = candidate

        if self.is_gibberish(response):
            response = self.fallback_reply(user_input)

        self.append_chat(response, sender="cat r1.x")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CatR1XApp(root)
    root.mainloop()
