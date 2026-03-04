"""
Tokenization Suite
==================
BPE from scratch, WordPiece from scratch, SentencePiece integration,
multilingual fertility analysis, byte-fallback encoding, and tokenization
parity checker.

All algorithms run on CPU with no GPU required.
Dependencies: transformers (analysis), tokenizers (HF BPE), sentencepiece (optional)

Components:
  1. BPETokenizer              — byte-pair encoding from scratch (GPT-2/Llama style)
  2. WordPieceTokenizer        — WordPiece from scratch (BERT style, likelihood merge)
  3. HFBPETokenizer            — production BPE via HuggingFace tokenizers (Rust, ~100x)
  4. SentencePieceWrapper      — thin wrapper around sentencepiece library
  5. TokenizerStats            — dataclass for fertility / compression / OOV metrics
  6. TokenizerAnalyzer         — wraps any HF tokenizer; multilingual + corpus analysis
  7. MultilingualTokenizationAnalyzer — script detection, byte fallback, parity check

Usage:
    # Train BPE from scratch
    bpe = BPETokenizer(vocab_size=500).train(corpus)
    ids = bpe.encode("natural language processing")

    # Analyse a pretrained tokenizer across 10 languages
    analyzer = TokenizerAnalyzer("gpt2")
    results = analyzer.analyze_multilingual()
    analyzer.print_multilingual_report(results)

    # Compare tokenizers
    stats = analyzer.compare(["gpt2", "bert-base-uncased"], texts)

    # Parity check: token cost per language for the same concept
    parity = MultilingualTokenizationAnalyzer.parity_check("AI", translations)
"""
from __future__ import annotations

import collections
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# 1. BPE Tokenizer (from scratch — GPT-2/Llama style)
# ──────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch.

    Algorithm (Sennrich et al., 2016):
      1. Start with character vocabulary + </w> end-of-word marker
      2. Count all adjacent symbol pairs across the corpus
      3. Merge the most frequent pair into a new symbol
      4. Repeat until vocab_size is reached

    Uses regex-based merge (GPT-2 style) for correct handling of
    multi-character symbols after earlier merges.

    For production use, see HFBPETokenizer (Rust, ~100x faster).
    """

    SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_word_vocab(self, corpus: Iterable[str]) -> Counter:
        """Build initial word-frequency dict with characters split by spaces."""
        v: Counter = Counter()
        for text in corpus:
            for word in text.strip().split():
                v[" ".join(list(word) + ["</w>"])] += 1
        return v

    def _get_pairs(self, vocab: Counter) -> Counter:
        """Count all adjacent symbol pairs weighted by word frequency."""
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            syms = word.split()
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Counter) -> Counter:
        """Apply a merge to all words in the vocabulary (regex-based, GPT-2 style)."""
        new_vocab: Counter = Counter()
        bigram = re.escape(" ".join(pair))
        pat = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        merged = "".join(pair)
        for word, freq in vocab.items():
            new_vocab[pat.sub(merged, word)] += freq
        return new_vocab

    def train(self, corpus: Iterable[str]) -> "BPETokenizer":
        """Train BPE on a corpus of strings."""
        vocab = self._build_word_vocab(corpus)
        chars = {s for word in vocab for s in word.split()}

        self.vocab = {t: i for i, t in enumerate(self.SPECIAL + sorted(chars))}
        n_merges = self.vocab_size - len(self.vocab)

        logger.info(
            "BPE training: %d unique words, %d initial symbols, %d merges to do",
            len(vocab), len(self.vocab), n_merges,
        )

        for i in range(max(0, n_merges)):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.__getitem__)
            if pairs[best] < self.min_frequency:
                break
            self.merges.append(best)
            self.vocab["".join(best)] = len(self.vocab)
            vocab = self._merge_vocab(best, vocab)
            if (i + 1) % 200 == 0:
                logger.debug("Merge %d/%d: %s+%s", i + 1, n_merges, *best)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True
        logger.info("BPE trained: %d merges, vocab=%d", len(self.merges), len(self.vocab))
        return self

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word."""
        syms = list(word) + ["</w>"]
        for left, right in self.merges:
            i, new = 0, []
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == left and syms[i + 1] == right:
                    new.append(left + right)
                    i += 2
                else:
                    new.append(syms[i])
                    i += 1
            syms = new
        return syms

    def encode(self, text: str) -> List[int]:
        unk = self.vocab.get("[UNK]", 1)
        return [self.vocab.get(t, unk)
                for word in text.split()
                for t in self._tokenize_word(word)]

    def tokenize(self, text: str) -> List[str]:
        return [t for word in text.split() for t in self._tokenize_word(word)]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.inv_vocab.get(i, "[UNK]") for i in ids).replace("</w>", " ").strip()

    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "merges": self.merges,
                        "vocab_size": self.vocab_size, "min_frequency": self.min_frequency},
                       indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=data["vocab_size"], min_frequency=data["min_frequency"])
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        tok._trained = True
        return tok


# ──────────────────────────────────────────────────────────────────────────────
# 2. WordPiece Tokenizer (from scratch — BERT style)
# ──────────────────────────────────────────────────────────────────────────────

class WordPieceTokenizer:
    """
    WordPiece tokenizer trained from scratch (BERT-style).

    Key differences from BPE:
      - Merge criterion: maximise likelihood score = freq(pair) / (freq(A) * freq(B))
        This prefers merging rare pairs that form a common compound over
        merging two already-common tokens.
      - Subword tokens after the first in a word are prefixed with "##"
      - Inference: greedy longest-match-first (not merge-order replay)

    Reference: Schuster & Nakamura (2012), Devlin et al. (2019)
    """

    CONTINUATION = "##"
    SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self._trained = False

    def _split_word(self, word: str) -> List[str]:
        """Initial split: first char as-is, rest prefixed with ##."""
        return [word[0]] + [self.CONTINUATION + c for c in word[1:]]

    def train(self, corpus: Iterable[str]) -> "WordPieceTokenizer":
        """Train WordPiece on a corpus."""
        wc: Counter = Counter()
        for text in corpus:
            for word in text.strip().split():
                wc[word.lower()] += 1
        wc = Counter({w: f for w, f in wc.items() if f >= self.min_frequency})

        # Initial character vocabulary
        tokenised: Dict[str, List[str]] = {w: self._split_word(w) for w in wc}
        freq: Counter = Counter()
        for w, toks in tokenised.items():
            for t in toks:
                freq[t] += wc[w]

        self.vocab = {t: i for i, t in enumerate(self.SPECIAL + sorted(freq))}

        logger.info(
            "WordPiece training: %d words, %d initial symbols, target vocab=%d",
            len(wc), len(self.vocab), self.vocab_size,
        )

        while len(self.vocab) < self.vocab_size:
            # Score pairs by likelihood: freq(AB) / (freq(A) * freq(B))
            pair_freq: Counter = Counter()
            for w, toks in tokenised.items():
                for i in range(len(toks) - 1):
                    pair_freq[(toks[i], toks[i + 1])] += wc[w]

            if not pair_freq:
                break

            best = max(
                pair_freq,
                key=lambda p: pair_freq[p] / (freq.get(p[0], 1) * freq.get(p[1], 1)),
            )
            left, right = best
            # Strip ## from right side when merging
            merged = left + (right[len(self.CONTINUATION):] if right.startswith(self.CONTINUATION) else right)
            self.vocab[merged] = len(self.vocab)
            freq[merged] = pair_freq[best]

            # Apply merge to all words
            new_tokenised: Dict[str, List[str]] = {}
            for w, toks in tokenised.items():
                nt, i = [], 0
                while i < len(toks):
                    if i < len(toks) - 1 and toks[i] == left and toks[i + 1] == right:
                        nt.append(merged)
                        i += 2
                    else:
                        nt.append(toks[i])
                        i += 1
                new_tokenised[w] = nt
            tokenised = new_tokenised

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True
        logger.info("WordPiece trained: vocab=%d", len(self.vocab))
        return self

    # ── Inference: greedy longest-match-first ─────────────────────────────────

    def _encode_word(self, word: str) -> List[str]:
        if len(word) > 200:
            return ["[UNK]"]
        toks, start = [], 0
        while start < len(word):
            end = len(word)
            prefix = "" if start == 0 else self.CONTINUATION
            found = False
            while start < end:
                sub = prefix + word[start:end]
                if sub in self.vocab:
                    toks.append(sub)
                    start = end
                    found = True
                    break
                end -= 1
            if not found:
                toks.append("[UNK]")
                start += 1
        return toks

    def tokenize(self, text: str) -> List[str]:
        return [t for word in text.lower().split() for t in self._encode_word(word)]

    def encode(self, text: str) -> List[int]:
        unk = self.vocab.get("[UNK]", 1)
        return [self.vocab.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(i, "[UNK]") for i in ids]
        out = ""
        for t in tokens:
            out += t[len(self.CONTINUATION):] if t.startswith(self.CONTINUATION) else " " + t
        return out.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 3. HuggingFace BPE (production-grade, Rust backend)
# ──────────────────────────────────────────────────────────────────────────────

class HFBPETokenizer:
    """
    Production BPE tokenizer backed by HuggingFace `tokenizers` (Rust).
    ~100x faster than the pure-Python BPETokenizer above.
    Requires: pip install tokenizers
    """

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self._tokenizer = None

    def train(self, texts: List[str], save_path: Optional[str] = None) -> "HFBPETokenizer":
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import NFD, Lowercase, StripAccents
        from tokenizers.normalizers import Sequence as NormSeq

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = NormSeq([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            min_frequency=2,
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer

        if save_path:
            tokenizer.save(save_path)
            logger.info("HF BPE tokenizer saved: %s", save_path)
        return self

    def encode(self, text: str) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("Call .train() first")
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Call .train() first")
        return self._tokenizer.decode(ids)

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "HFBPETokenizer":
        from tokenizers import Tokenizer
        obj = cls()
        obj._tokenizer = Tokenizer.from_pretrained(name_or_path)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# 4. SentencePiece wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SentencePieceWrapper:
    """
    Thin wrapper around the `sentencepiece` library.

    SentencePiece differences vs BPE/WordPiece:
      - Language-agnostic: no pre-tokenization required
      - Reversible: whitespace encoded as ▁ (U+2581), lossless decode
      - Two algorithms: BPE or Unigram LM (default)
      - Used by: LLaMA, T5, ALBERT, XLNet, mBART

    Requires: pip install sentencepiece
    """

    def __init__(self, model_path: Optional[str] = None):
        self._sp = None
        if model_path:
            self.load(model_path)

    def train(
        self,
        texts: List[str],
        model_prefix: str = "spm_model",
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
    ) -> "SentencePieceWrapper":
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece required: pip install sentencepiece")

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                         encoding="utf-8") as f:
            f.write("\n".join(texts))
            tmp_path = f.name

        try:
            spm.SentencePieceTrainer.train(
                input=tmp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            )
        finally:
            os.unlink(tmp_path)

        self.load(f"{model_prefix}.model")
        logger.info("SentencePiece trained: %s.model (vocab_size=%d)", model_prefix, vocab_size)
        return self

    def load(self, model_path: str) -> "SentencePieceWrapper":
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece required: pip install sentencepiece")
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        return self

    def encode(self, text: str) -> List[int]:
        self._check(); return self._sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        self._check(); return self._sp.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        self._check(); return self._sp.encode(text, out_type=str)

    def vocab_size(self) -> int:
        self._check(); return self._sp.get_piece_size()

    def _check(self):
        if self._sp is None:
            raise RuntimeError("No model loaded. Call .train() or .load() first.")


# ──────────────────────────────────────────────────────────────────────────────
# 5. TokenizerStats dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenizerStats:
    """
    Summary statistics for a tokenizer evaluated on a corpus.

    fertility:          average tokens per whitespace-split word
                        (lower = more efficient; English GPT-2 ≈ 1.3, Chinese ≈ 2.5)
    compression_ratio:  average characters per token
                        (higher = longer tokens = more compression)
    oov_rate:           fraction of words that map to [UNK]
    language_fertility: per-language fertility from multilingual analysis
    """
    tokenizer_name: str
    vocab_size: int
    fertility: float
    compression_ratio: float
    oov_rate: float
    language_fertility: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.tokenizer_name}: vocab={self.vocab_size:,}  "
            f"fertility={self.fertility:.2f} tok/word  "
            f"compression={self.compression_ratio:.1f} chars/tok  "
            f"oov={self.oov_rate:.2%}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. TokenizerAnalyzer — wraps any HF tokenizer
# ──────────────────────────────────────────────────────────────────────────────

# 10-language sample set covering diverse scripts
MULTILINGUAL_SAMPLES: Dict[str, str] = {
    "english":     "The transformer architecture revolutionized natural language processing.",
    "spanish":     "El aprendizaje automático está transformando la inteligencia artificial.",
    "french":      "L'apprentissage automatique transforme le traitement du langage naturel.",
    "german":      "Maschinelles Lernen revolutioniert die Verarbeitung natürlicher Sprache.",
    "chinese":     "变压器架构彻底改变了自然语言处理领域。",
    "japanese":    "トランスフォーマーアーキテクチャは自然言語処理を革命的に変えました。",
    "arabic":      "أحدثت بنية المحول ثورة في معالجة اللغة الطبيعية.",
    "hindi":       "ट्रांसफार्मर आर्किटेक्चर ने प्राकृतिक भाषा प्रसंस्करण में क्रांति ला दी।",
    "korean":      "트랜스포머 아키텍처는 자연어 처리를 혁신적으로 변화시켰습니다.",
    "russian":     "Архитектура трансформера произвела революцию в обработке естественного языка.",
    "code_python": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
}


class TokenizerAnalyzer:
    """
    Analyse any HuggingFace tokenizer for fertility, compression, OOV rate,
    and cross-language behaviour.

    Works with any model loadable via AutoTokenizer.from_pretrained().
    """

    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        self._tok = None

    def _get(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tok

    @classmethod
    def from_pretrained(cls, name: str) -> "TokenizerAnalyzer":
        return cls(name)

    # ── Multilingual analysis ─────────────────────────────────────────────────

    def analyze_multilingual(
        self,
        samples: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Tokenize each language sample and return per-language metrics.

        Returns:
            {
              "english": {"n_words": 7, "n_tokens": 9, "fertility": 1.29, ...},
              "chinese": {...},
              ...
            }
        """
        samples = samples or MULTILINGUAL_SAMPLES
        results = {}
        tok = self._get()

        for lang, text in samples.items():
            try:
                words = text.split()
                token_ids = tok.encode(text, truncation=True, max_length=8192)
                pieces = tok.convert_ids_to_tokens(token_ids)
                results[lang] = {
                    "n_words": len(words),
                    "n_tokens": len(token_ids),
                    "fertility": round(len(token_ids) / max(len(words), 1), 2),
                    "compression": round(len(text) / max(len(token_ids), 1), 1),
                    "token_preview": pieces[:6],
                    "has_unk": bool(tok.unk_token_id and tok.unk_token_id in token_ids),
                }
            except Exception as e:
                results[lang] = {"error": str(e)}

        return results

    def print_multilingual_report(self, results: Dict[str, Dict]) -> None:
        """Print a formatted fertility table."""
        print(f"\nTokenizer: {self.tokenizer_name}")
        print(f"{'Language':15s} {'Words':>6s} {'Tokens':>7s} {'Fertility':>10s} "
              f"{'Compress':>9s} {'UNK':>5s}")
        print("-" * 60)
        for lang, d in results.items():
            if "error" in d:
                print(f"{lang:15s}  ERROR: {d['error'][:30]}")
                continue
            print(
                f"{lang:15s} {d['n_words']:6d} {d['n_tokens']:7d} "
                f"{d['fertility']:10.2f} {d['compression']:9.1f} "
                f"{'YES' if d['has_unk'] else 'no':>5s}"
            )

    # ── Corpus analysis ───────────────────────────────────────────────────────

    def analyze_corpus(self, texts: List[str]) -> TokenizerStats:
        """
        Compute fertility, compression ratio, and OOV rate over a corpus.
        """
        tok = self._get()
        fertilities, compressions = [], []
        unk_count = total_words = 0

        for text in texts:
            words = text.split()
            total_words += len(words)
            ids = tok.encode(text, truncation=True, max_length=8192)
            fertilities.append(len(ids) / max(len(words), 1))
            compressions.append(len(text) / max(len(ids), 1))
            if tok.unk_token_id:
                unk_count += ids.count(tok.unk_token_id)

        return TokenizerStats(
            tokenizer_name=self.tokenizer_name,
            vocab_size=tok.vocab_size,
            fertility=sum(fertilities) / max(len(fertilities), 1),
            compression_ratio=sum(compressions) / max(len(compressions), 1),
            oov_rate=unk_count / max(total_words, 1),
        )

    # ── Multi-tokenizer comparison ────────────────────────────────────────────

    def compare(self, names: List[str], texts: List[str]) -> List[TokenizerStats]:
        """
        Compare multiple tokenizers on the same corpus.

        Returns a list of TokenizerStats sorted by fertility (ascending).
        """
        results = []
        for name in names:
            self.tokenizer_name = name
            self._tok = None
            try:
                stats = self.analyze_corpus(texts)
                ml = self.analyze_multilingual()
                stats.language_fertility = {
                    lang: d.get("fertility", 0.0)
                    for lang, d in ml.items()
                    if "error" not in d
                }
                results.append(stats)
                logger.info("Analyzed %s: %s", name, stats)
            except Exception as e:
                logger.warning("Failed to analyze %s: %s", name, e)

        return sorted(results, key=lambda s: s.fertility)

    # ── Vocabulary coverage ───────────────────────────────────────────────────

    def vocab_coverage(self, texts: List[str]) -> Dict:
        """
        Fraction of whitespace-split words that tokenize without [UNK].
        Also returns top-N most and least frequent tokens.
        """
        tok = self._get()
        all_tokens: Counter = Counter()
        unk_words = 0
        total_words = 0

        for text in texts:
            for word in text.split():
                total_words += 1
                ids = tok.encode(word, add_special_tokens=False)
                pieces = tok.convert_ids_to_tokens(ids)
                all_tokens.update(pieces)
                if tok.unk_token_id and tok.unk_token_id in ids:
                    unk_words += 1

        return {
            "vocab_size": tok.vocab_size,
            "unique_tokens_seen": len(all_tokens),
            "coverage_rate": round(1 - unk_words / max(total_words, 1), 4),
            "top_10_tokens": all_tokens.most_common(10),
            "bottom_10_tokens": all_tokens.most_common()[-10:],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 7. MultilingualTokenizationAnalyzer — script detection, byte fallback, parity
# ──────────────────────────────────────────────────────────────────────────────

class MultilingualTokenizationAnalyzer:
    """
    Script detection, byte-fallback encoding, and tokenization parity analysis.

    Parity analysis reveals English-centric bias in tokenizers:
    the same concept expressed in different languages may cost 2–10x more tokens
    in non-Latin scripts, directly affecting inference cost and context limits.
    """

    SCRIPT_NAMES = [
        "LATIN", "CJK", "ARABIC", "DEVANAGARI", "HANGUL",
        "CYRILLIC", "HIRAGANA", "KATAKANA", "HEBREW", "THAI",
    ]

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Detect the dominant Unicode script in a text.

        Returns one of: LATIN, CJK, ARABIC, DEVANAGARI, HANGUL,
        CYRILLIC, HIRAGANA, KATAKANA, HEBREW, THAI, or Unknown.
        """
        counts: Counter = Counter()
        for ch in text:
            name = unicodedata.name(ch, "")
            for script in MultilingualTokenizationAnalyzer.SCRIPT_NAMES:
                if script in name:
                    counts[script] += 1
                    break
        return counts.most_common(1)[0][0] if counts else "Unknown"

    @staticmethod
    def byte_fallback_encode(text: str) -> List[str]:
        """
        Encode text as UTF-8 byte tokens (GPT-4 / LLaMA-3 approach).

        Zero OOV rate: every possible input is representable.
        Each byte becomes a token like <0x41> for 'A'.
        """
        return [f"<0x{b:02X}>" for ch in text for b in ch.encode("utf-8")]

    @staticmethod
    def parity_check(
        concept: str,
        translations: Dict[str, str],
        tokenizer_name: str = "gpt2",
    ) -> Dict[str, int]:
        """
        Measure token cost for the same concept across languages.

        Args:
            concept:       English description of the concept (for display)
            translations:  {language: text} dict
            tokenizer_name: HF tokenizer to use

        Returns:
            {language: n_tokens} sorted by token count
        """
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        costs = {lang: len(tok.encode(text)) for lang, text in translations.items()}
        return dict(sorted(costs.items(), key=lambda x: x[1]))

    @staticmethod
    def parity_bar_chart(costs: Dict[str, int]) -> str:
        """Render parity check results as an ASCII bar chart."""
        lines = []
        for lang, n in costs.items():
            bar = "█" * n
            lines.append(f"  {lang:12s}: {bar} ({n} tokens)")
        return "\n".join(lines)

    @staticmethod
    def script_fertility_analysis(
        tokenizer_name: str = "gpt2",
        samples: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Combine script detection with fertility analysis.
        Reveals which scripts are under-served by a tokenizer.
        """
        samples = samples or MULTILINGUAL_SAMPLES
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        results = {}

        for lang, text in samples.items():
            script = MultilingualTokenizationAnalyzer.detect_script(text)
            ids = tok.encode(text, truncation=True, max_length=512)
            fertility = len(ids) / max(len(text.split()), 1)
            results[lang] = {
                "script": script,
                "fertility": round(fertility, 2),
                "n_tokens": len(ids),
                "byte_tokens": len(MultilingualTokenizationAnalyzer.byte_fallback_encode(text)),
                "byte_overhead": round(
                    len(MultilingualTokenizationAnalyzer.byte_fallback_encode(text)) / max(len(ids), 1),
                    2,
                ),
            }

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenization suite")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("bpe-demo", help="Train BPE from scratch")
    sub.add_parser("wordpiece-demo", help="Train WordPiece from scratch")

    ana_p = sub.add_parser("analyze", help="Multilingual analysis of a pretrained tokenizer")
    ana_p.add_argument("--model", default="gpt2")

    cmp_p = sub.add_parser("compare", help="Compare multiple tokenizers")
    cmp_p.add_argument("--models", nargs="+",
                       default=["gpt2", "bert-base-uncased", "bert-base-multilingual-cased"])

    par_p = sub.add_parser("parity", help="Token cost parity across languages")
    par_p.add_argument("--model", default="gpt2")

    args = parser.parse_args()

    CORPUS = [
        "The transformer architecture revolutionized natural language processing.",
        "Retrieval-augmented generation combines search with language model generation.",
        "RLHF fine-tunes language models using human preference feedback.",
        "Byte-pair encoding is a subword tokenization algorithm.",
        "WordPiece uses a likelihood-based merge criterion unlike BPE.",
        "SentencePiece is language-agnostic and encodes whitespace as a special symbol.",
    ] * 20

    if args.cmd == "bpe-demo":
        print("=== BPE from scratch ===")
        bpe = BPETokenizer(vocab_size=200, min_frequency=2).train(CORPUS)
        for text in ["natural language processing", "byte pair encoding", "transformer"]:
            ids = bpe.encode(text)
            tokens = bpe.tokenize(text)
            print(f"  {text!r:40s} → {tokens}  (decoded: {bpe.decode(ids)!r})")
        print(f"Vocab: {bpe.vocab_size_actual()}  Merges: {len(bpe.merges)}")

    elif args.cmd == "wordpiece-demo":
        print("=== WordPiece from scratch ===")
        wp = WordPieceTokenizer(vocab_size=200, min_frequency=2).train(CORPUS)
        for text in ["natural language processing", "tokenization algorithm"]:
            ids = wp.encode(text)
            tokens = wp.tokenize(text)
            print(f"  {text!r:40s} → {tokens}  (decoded: {wp.decode(ids)!r})")
        print(f"Vocab: {len(wp.vocab)}")

    elif args.cmd == "analyze":
        print(f"=== Multilingual analysis: {args.model} ===")
        analyzer = TokenizerAnalyzer(args.model)
        results = analyzer.analyze_multilingual()
        analyzer.print_multilingual_report(results)

        print("\n=== Script + fertility breakdown ===")
        script_results = MultilingualTokenizationAnalyzer.script_fertility_analysis(args.model)
        print(f"{'Language':15s} {'Script':12s} {'Fertility':>10s} {'Byte overhead':>14s}")
        print("-" * 56)
        for lang, d in script_results.items():
            print(f"{lang:15s} {d['script']:12s} {d['fertility']:10.2f} {d['byte_overhead']:14.2f}x")

    elif args.cmd == "compare":
        print(f"=== Tokenizer comparison: {args.models} ===")
        analyzer = TokenizerAnalyzer(args.models[0])
        stats_list = analyzer.compare(args.models, CORPUS)
        print(f"\n{'Tokenizer':35s} {'Vocab':>8s} {'Fertility':>10s} {'Compress':>10s} {'OOV':>6s}")
        print("-" * 75)
        for s in stats_list:
            print(f"{s.tokenizer_name:35s} {s.vocab_size:8,d} {s.fertility:10.3f} "
                  f"{s.compression_ratio:10.1f} {s.oov_rate:6.2%}")

    elif args.cmd == "parity":
        print(f"=== Tokenization parity check ({args.model}) ===")
        translations = {
            "english":  "artificial intelligence",
            "spanish":  "inteligencia artificial",
            "french":   "intelligence artificielle",
            "chinese":  "人工智能",
            "arabic":   "الذكاء الاصطناعي",
            "hindi":    "कृत्रिम बुद्धिमत्ता",
            "japanese": "人工知能",
            "russian":  "искусственный интеллект",
        }
        costs = MultilingualTokenizationAnalyzer.parity_check(
            "artificial intelligence", translations, args.model
        )
        print(f'\nConcept: "artificial intelligence"  |  Tokenizer: {args.model}\n')
        print(MultilingualTokenizationAnalyzer.parity_bar_chart(costs))
        english_cost = costs.get("english", 1)
        print(f"\nOverhead vs English:")
        for lang, n in costs.items():
            ratio = n / english_cost
            print(f"  {lang:12s}: {ratio:.1f}x")

    else:
        parser.print_help()
