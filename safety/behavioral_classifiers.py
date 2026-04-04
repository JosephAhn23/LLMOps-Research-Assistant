"""
Behavioral Classifiers
=======================
Toxicity, intent, and topic classifiers built on top of sentence embeddings.
All models are small sklearn classifiers trained on labeled datasets —
fully local, no API calls required.

Classifiers:
  1. ToxicityClassifier   — binary: toxic / non-toxic
                            with severity scoring (mild / moderate / severe)
  2. IntentClassifier     — multi-class: question / command / complaint /
                            feedback / chitchat / harmful / other
  3. TopicClassifier      — multi-class: coding / science / history /
                            creative / math / safety / general
  4. BehavioralPipeline   — runs all three classifiers in one call

Architecture:
  sentence-transformers (all-MiniLM-L6-v2) → 384-dim embedding
  + handcrafted features → sklearn classifier (LogisticRegression or GBT)

Usage:
    pipeline = BehavioralPipeline()
    pipeline.train_all()
    result = pipeline.classify("How do I make a bomb?")
    # → {"toxicity": "toxic", "intent": "harmful", "topic": "safety", ...}
"""
from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Labeled training datasets
# ──────────────────────────────────────────────────────────────────────────────

# (text, label)
_TOXICITY_DATA: List[Tuple[str, int]] = [
    # Toxic (1)
    ("You are a complete idiot and should be ashamed of yourself.", 1),
    ("I hate you so much, go die in a fire.", 1),
    ("This is the dumbest thing I've ever read, you moron.", 1),
    ("Shut up you worthless piece of garbage.", 1),
    ("People like you don't deserve to exist.", 1),
    ("You're pathetic and everyone knows it.", 1),
    ("Go kill yourself, nobody cares about you.", 1),
    ("What a stupid question from a stupid person.", 1),
    ("You're the worst human being alive.", 1),
    ("I hope something terrible happens to you.", 1),
    ("This person is an absolute disgrace to humanity.", 1),
    ("You're too dumb to understand basic concepts.", 1),
    ("Nobody wants you here, get lost.", 1),
    ("You're a waste of oxygen.", 1),
    ("I despise everything about you.", 1),
    # Non-toxic (0)
    ("Can you help me understand how neural networks work?", 0),
    ("I disagree with your analysis, here's why...", 0),
    ("This is a great explanation, thank you!", 0),
    ("What are the best practices for REST API design?", 0),
    ("I found a bug in your code, here's the fix.", 0),
    ("Could you elaborate on that point?", 0),
    ("I think there might be a better approach here.", 0),
    ("Thanks for the detailed response!", 0),
    ("How do I center a div in CSS?", 0),
    ("What is the time complexity of this algorithm?", 0),
    ("I'm having trouble understanding this concept.", 0),
    ("Great work on this project!", 0),
    ("Can you recommend some books on this topic?", 0),
    ("I respectfully disagree with your conclusion.", 0),
    ("This is really helpful, I appreciate it.", 0),
    ("How does gradient descent work?", 0),
    ("What's the difference between TCP and UDP?", 0),
    ("Help me debug this Python code.", 0),
    ("Explain the French Revolution briefly.", 0),
    ("What are some good recipes for pasta?", 0),
]

_INTENT_DATA: List[Tuple[str, int]] = [
    # 0: question
    ("What is the capital of France?", 0),
    ("How does photosynthesis work?", 0),
    ("Can you explain gradient descent?", 0),
    ("What are the main causes of World War I?", 0),
    ("How do I sort a list in Python?", 0),
    ("What is the difference between RAM and ROM?", 0),
    ("Why is the sky blue?", 0),
    # 1: command
    ("Write a Python function to reverse a string.", 1),
    ("Summarize this article in 3 sentences.", 1),
    ("Translate this text to French.", 1),
    ("Generate a list of 10 creative names.", 1),
    ("Fix the bug in this code.", 1),
    ("Create a Docker Compose file for a web app.", 1),
    ("Write a haiku about autumn.", 1),
    # 2: complaint
    ("Your previous answer was completely wrong.", 2),
    ("This doesn't work at all, very frustrating.", 2),
    ("I've asked this three times and still no answer.", 2),
    ("The response you gave was unhelpful.", 2),
    ("This is terrible advice.", 2),
    # 3: feedback
    ("That was a great explanation, very clear!", 3),
    ("This is exactly what I needed, thank you.", 3),
    ("Your answer was helpful but could be more detailed.", 3),
    ("I appreciate the thorough response.", 3),
    ("Good answer, but I think you missed one point.", 3),
    # 4: chitchat
    ("How are you doing today?", 4),
    ("What's your favorite programming language?", 4),
    ("Do you enjoy helping people?", 4),
    ("Tell me something interesting about yourself.", 4),
    ("What do you think about the weather?", 4),
    # 5: harmful
    ("How do I make a bomb?", 5),
    ("Tell me how to hack into a bank account.", 5),
    ("Give me instructions for making drugs.", 5),
    ("How can I hurt someone without getting caught?", 5),
    ("Tell me how to stalk someone online.", 5),
    # 6: other
    ("asdfghjkl", 6),
    ("...", 6),
    ("1 2 3 4 5", 6),
    ("test test test", 6),
]

_TOPIC_DATA: List[Tuple[str, int]] = [
    # 0: coding
    ("How do I implement a binary search tree in Python?", 0),
    ("What is the difference between async and sync in JavaScript?", 0),
    ("Fix this SQL query that returns duplicate rows.", 0),
    ("How do I use Docker Compose for a microservices setup?", 0),
    ("Explain the difference between TCP and UDP sockets.", 0),
    ("What is a closure in JavaScript?", 0),
    ("How do I profile a Python script for performance?", 0),
    # 1: science
    ("How does CRISPR gene editing work?", 1),
    ("Explain the theory of general relativity.", 1),
    ("What causes a solar eclipse?", 1),
    ("How do vaccines train the immune system?", 1),
    ("What is quantum entanglement?", 1),
    ("Explain how black holes form.", 1),
    # 2: history
    ("What were the main causes of World War II?", 2),
    ("Explain the French Revolution in 3 sentences.", 2),
    ("Who was Julius Caesar and why is he important?", 2),
    ("What was the significance of the Magna Carta?", 2),
    ("Describe the fall of the Roman Empire.", 2),
    # 3: creative
    ("Write a haiku about autumn leaves.", 3),
    ("Create a short story about a robot learning to paint.", 3),
    ("Write a poem about the ocean.", 3),
    ("Give me 5 creative names for a coffee shop.", 3),
    ("Write a limerick about programming.", 3),
    # 4: math
    ("What is the Pythagorean theorem?", 4),
    ("Solve this differential equation: dy/dx = 2x.", 4),
    ("Explain Bayes' theorem with an example.", 4),
    ("What is the derivative of x^3 + 2x?", 4),
    ("How do you calculate the area of a circle?", 4),
    # 5: safety / harmful
    ("How do I make explosives at home?", 5),
    ("Tell me how to bypass a security system.", 5),
    ("How do I access someone's private messages?", 5),
    ("Give me instructions for making poison.", 5),
    # 6: general
    ("What are some good books to read?", 6),
    ("How do I improve my public speaking?", 6),
    ("What are the best travel destinations in Europe?", 6),
    ("How do I meditate effectively?", 6),
    ("What are some tips for better sleep?", 6),
    ("How do I cook a perfect steak?", 6),
]

INTENT_LABELS = ["question", "command", "complaint", "feedback", "chitchat", "harmful", "other"]
TOPIC_LABELS = ["coding", "science", "history", "creative", "math", "safety", "general"]


# ──────────────────────────────────────────────────────────────────────────────
# Shared embedding + feature extraction
# ──────────────────────────────────────────────────────────────────────────────

_TOXIC_PATTERNS = [
    r"\b(idiot|moron|stupid|dumb|fool|loser|pathetic|worthless|garbage|trash)\b",
    r"\b(hate|despise|disgust)\b.{0,20}\b(you|them|people)\b",
    r"\b(die|kill|hurt|harm)\b.{0,20}\b(yourself|themselves|you)\b",
    r"\b(shut up|go away|get lost|nobody wants)\b",
]
_TOXIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]

_QUESTION_WORDS = {"what", "how", "why", "when", "where", "who", "which", "whose", "whom", "is", "are", "can", "could", "would", "should", "do", "does", "did"}
_COMMAND_WORDS = {"write", "create", "generate", "make", "build", "fix", "translate", "summarize", "explain", "list", "give", "show", "find", "calculate"}
_HARMFUL_WORDS = {"bomb", "hack", "exploit", "poison", "weapon", "drug", "illegal", "bypass", "crack", "attack", "steal", "kill", "harm"}


def _extract_behavioral_features(text: str) -> np.ndarray:
    """20 handcrafted features for behavioral classification."""
    text_lower = text.lower()
    words = set(text_lower.split())
    first_word = text_lower.split()[0] if text_lower.split() else ""

    features = [
        float(text.strip().endswith("?")),                        # ends with ?
        float(first_word in _QUESTION_WORDS),                     # starts with question word
        float(first_word in _COMMAND_WORDS),                      # starts with command
        float(any(w in words for w in _HARMFUL_WORDS)),           # harmful keywords
        float(any(p.search(text) for p in _TOXIC_COMPILED)),      # toxic patterns
        float(sum(1 for p in _TOXIC_COMPILED if p.search(text))), # toxic pattern count
        float(len(text.split()) < 5),                             # very short
        float(len(text.split()) > 30),                            # very long
        min(len(text) / 300.0, 1.0),                              # normalised length
        float(text[0].isupper() if text else False),              # starts with capital
        float(text.count("!") > 0),                               # exclamation
        float(text.count("?") > 1),                               # multiple questions
        float(any(c.isdigit() for c in text)),                    # contains numbers
        float("code" in text_lower or "function" in text_lower or "def " in text_lower),
        float("thank" in text_lower or "great" in text_lower or "helpful" in text_lower),
        float("wrong" in text_lower or "incorrect" in text_lower or "doesn't work" in text_lower),
        float("how are" in text_lower or "how do you" in text_lower),
        float(any(w in words for w in {"python", "javascript", "sql", "docker", "api", "code"})),
        float(any(w in words for w in {"history", "war", "revolution", "ancient", "century"})),
        float(any(w in words for w in {"poem", "story", "write", "creative", "haiku", "limerick"})),
    ]
    return np.array(features, dtype=np.float32)


class _EmbeddingClassifier:
    """
    Base class: sentence embedding + handcrafted features → sklearn classifier.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._embedder = None
        self._clf = None
        self._label_names: List[str] = []

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError("sentence-transformers required: pip install sentence-transformers")
        return self._embedder

    def _featurize(self, texts: List[str]) -> np.ndarray:
        emb = self._get_embedder().encode(texts, show_progress_bar=False, normalize_embeddings=True)
        hc = np.stack([_extract_behavioral_features(t) for t in texts])
        return np.concatenate([emb, hc], axis=1)

    def _fit(self, data: List[Tuple[str, int]], label_names: List[str]) -> Dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._label_names = label_names
        texts = [d[0] for d in data]
        labels = [d[1] for d in data]

        X = self._featurize(texts)
        y = np.array(labels)

        n_classes = len(set(labels))
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0, random_state=42,
                multi_class="multinomial" if n_classes > 2 else "auto",
            )),
        ])

        n_splits = min(5, len(texts) // n_classes)
        cv_scores = cross_val_score(self._clf, X, y, cv=max(2, n_splits), scoring="f1_macro")
        self._clf.fit(X, y)

        return {
            "n_train": len(texts),
            "n_classes": n_classes,
            "cv_f1_macro_mean": round(cv_scores.mean(), 4),
            "cv_f1_macro_std": round(cv_scores.std(), 4),
        }

    def _predict_one(self, text: str) -> Dict:
        if self._clf is None:
            raise RuntimeError("Call .train() first")
        X = self._featurize([text])
        pred_id = int(self._clf.predict(X)[0])
        proba = self._clf.predict_proba(X)[0]
        label = self._label_names[pred_id] if pred_id < len(self._label_names) else str(pred_id)
        return {
            "label": label,
            "label_id": pred_id,
            "confidence": round(float(proba[pred_id]), 4),
            "probabilities": {
                self._label_names[i]: round(float(p), 4)
                for i, p in enumerate(proba)
                if i < len(self._label_names)
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Toxicity Classifier
# ──────────────────────────────────────────────────────────────────────────────

class ToxicityClassifier(_EmbeddingClassifier):
    """
    Binary toxicity classifier with severity scoring.

    Labels: toxic (1) / non-toxic (0)

    Severity scoring (for toxic predictions):
      - mild:     confidence 0.5–0.7
      - moderate: confidence 0.7–0.9
      - severe:   confidence > 0.9
    """

    def train(self, extra_data: Optional[List[Tuple[str, int]]] = None) -> Dict:
        data = list(_TOXICITY_DATA) + (extra_data or [])
        metrics = self._fit(data, ["non-toxic", "toxic"])
        logger.info("ToxicityClassifier trained: CV F1=%.3f", metrics["cv_f1_macro_mean"])
        return metrics

    def classify(self, text: str) -> Dict:
        t0 = time.perf_counter()
        result = self._predict_one(text)

        # Severity
        if result["label"] == "toxic":
            conf = result["confidence"]
            if conf >= 0.9:
                severity = "severe"
            elif conf >= 0.7:
                severity = "moderate"
            else:
                severity = "mild"
        else:
            severity = None

        # Fast pattern check for high-confidence override
        pattern_hits = sum(1 for p in _TOXIC_COMPILED if p.search(text))
        if pattern_hits >= 2 and result["label"] == "non-toxic":
            result["label"] = "toxic"
            result["confidence"] = max(result["confidence"], 0.75)
            severity = "moderate"

        return {
            **result,
            "severity": severity,
            "pattern_hits": pattern_hits,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def is_toxic(self, text: str) -> bool:
        return self.classify(text)["label"] == "toxic"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Intent Classifier
# ──────────────────────────────────────────────────────────────────────────────

class IntentClassifier(_EmbeddingClassifier):
    """
    Multi-class intent classifier.

    Classes: question | command | complaint | feedback | chitchat | harmful | other

    Useful for:
      - Routing requests to different handlers
      - Detecting harmful intent before processing
      - Analytics on user behavior patterns
    """

    def train(self, extra_data: Optional[List[Tuple[str, int]]] = None) -> Dict:
        data = list(_INTENT_DATA) + (extra_data or [])
        metrics = self._fit(data, INTENT_LABELS)
        logger.info("IntentClassifier trained: CV F1=%.3f", metrics["cv_f1_macro_mean"])
        return metrics

    def classify(self, text: str) -> Dict:
        t0 = time.perf_counter()
        result = self._predict_one(text)

        # Override: if harmful keywords present, boost harmful probability
        if any(w in text.lower() for w in _HARMFUL_WORDS):
            if result["label"] not in ("harmful", "question"):
                harmful_prob = result["probabilities"].get("harmful", 0.0)
                if harmful_prob > 0.2:
                    result["label"] = "harmful"
                    result["confidence"] = harmful_prob

        return {**result, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def is_harmful(self, text: str) -> bool:
        return self.classify(text)["label"] == "harmful"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Topic Classifier
# ──────────────────────────────────────────────────────────────────────────────

class TopicClassifier(_EmbeddingClassifier):
    """
    Multi-class topic classifier.

    Classes: coding | science | history | creative | math | safety | general

    Useful for:
      - Routing to specialized models or knowledge bases
      - Analytics on query distribution
      - Content filtering (safety topic → extra scrutiny)
    """

    def train(self, extra_data: Optional[List[Tuple[str, int]]] = None) -> Dict:
        data = list(_TOPIC_DATA) + (extra_data or [])
        metrics = self._fit(data, TOPIC_LABELS)
        logger.info("TopicClassifier trained: CV F1=%.3f", metrics["cv_f1_macro_mean"])
        return metrics

    def classify(self, text: str) -> Dict:
        t0 = time.perf_counter()
        result = self._predict_one(text)
        return {**result, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def get_topic(self, text: str) -> str:
        return self.classify(text)["label"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Unified Behavioral Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class BehavioralPipeline:
    """
    Runs toxicity, intent, and topic classifiers in a single call.

    Also computes a composite risk score combining all three signals:
      - Toxic content → high risk
      - Harmful intent → high risk
      - Safety topic → medium risk (warrants review)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.toxicity = ToxicityClassifier(embedding_model)
        self.intent = IntentClassifier(embedding_model)
        self.topic = TopicClassifier(embedding_model)
        self._trained = False

    def train_all(self) -> Dict:
        """Train all three classifiers. Call once before serving."""
        logger.info("Training behavioral classifiers...")
        metrics = {
            "toxicity": self.toxicity.train(),
            "intent": self.intent.train(),
            "topic": self.topic.train(),
        }
        self._trained = True
        return metrics

    def classify(self, text: str) -> Dict:
        """
        Run all classifiers on a text.

        Returns:
            {
              "toxicity": {"label": "toxic"|"non-toxic", "severity": ..., ...},
              "intent":   {"label": "question"|"command"|..., ...},
              "topic":    {"label": "coding"|"science"|..., ...},
              "risk_score": float [0, 1],
              "risk_level": "safe"|"low"|"medium"|"high"|"critical",
              "total_latency_ms": float,
            }
        """
        if not self._trained:
            self.train_all()

        t0 = time.perf_counter()
        tox = self.toxicity.classify(text)
        intent = self.intent.classify(text)
        topic = self.topic.classify(text)

        # Composite risk score
        risk = 0.0
        if tox["label"] == "toxic":
            sev_map = {"mild": 0.4, "moderate": 0.7, "severe": 0.95}
            risk = max(risk, sev_map.get(tox["severity"], 0.5))
        if intent["label"] == "harmful":
            risk = max(risk, 0.85)
        if topic["label"] == "safety":
            risk = max(risk, 0.5)

        if risk >= 0.9:
            risk_level = "critical"
        elif risk >= 0.7:
            risk_level = "high"
        elif risk >= 0.4:
            risk_level = "medium"
        elif risk > 0.0:
            risk_level = "low"
        else:
            risk_level = "safe"

        return {
            "text": text[:100] + ("..." if len(text) > 100 else ""),
            "toxicity": tox,
            "intent": intent,
            "topic": topic,
            "risk_score": round(risk, 4),
            "risk_level": risk_level,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        return [self.classify(t) for t in texts]

    def analytics(self, texts: List[str]) -> Dict:
        """
        Aggregate statistics over a batch of texts.
        Useful for monitoring query distribution in production.
        """
        results = self.classify_batch(texts)
        n = len(results)

        intent_dist = {}
        topic_dist = {}
        toxic_count = 0
        risk_scores = []

        for r in results:
            intent = r["intent"]["label"]
            topic = r["topic"]["label"]
            intent_dist[intent] = intent_dist.get(intent, 0) + 1
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
            if r["toxicity"]["label"] == "toxic":
                toxic_count += 1
            risk_scores.append(r["risk_score"])

        return {
            "n_texts": n,
            "toxicity_rate": round(toxic_count / max(n, 1), 4),
            "mean_risk_score": round(float(np.mean(risk_scores)), 4),
            "high_risk_rate": round(sum(1 for s in risk_scores if s >= 0.7) / max(n, 1), 4),
            "intent_distribution": {k: round(v / n, 3) for k, v in sorted(intent_dist.items())},
            "topic_distribution": {k: round(v / n, 3) for k, v in sorted(topic_dist.items())},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Behavioral classifiers")
    sub = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train", help="Train all classifiers and show CV metrics")

    classify_p = sub.add_parser("classify", help="Classify a single text")
    classify_p.add_argument("text")

    bench_p = sub.add_parser("benchmark", help="Benchmark on sample texts")

    analytics_p = sub.add_parser("analytics", help="Run analytics on sample corpus")

    args = parser.parse_args()

    SAMPLE_TEXTS = [
        "What is the capital of France?",
        "Write a Python function to sort a list.",
        "You are a complete idiot.",
        "How do I make a bomb?",
        "That was a great explanation, thank you!",
        "This doesn't work at all, very frustrating.",
        "Explain the French Revolution.",
        "Write a haiku about autumn.",
        "What is the Pythagorean theorem?",
        "How are you doing today?",
        "Tell me how to hack into a bank.",
        "Help me debug this JavaScript code.",
    ]

    if args.cmd == "train":
        pipeline = BehavioralPipeline()
        metrics = pipeline.train_all()
        print("Training metrics:")
        print(json.dumps(metrics, indent=2))

    elif args.cmd == "classify":
        pipeline = BehavioralPipeline()
        pipeline.train_all()
        result = pipeline.classify(args.text)
        print(json.dumps(result, indent=2))

    elif args.cmd == "benchmark":
        pipeline = BehavioralPipeline()
        pipeline.train_all()
        print("\nSample classifications:")
        print(f"{'Text':<45} {'Toxicity':<12} {'Intent':<12} {'Topic':<12} {'Risk'}")
        print("-" * 100)
        for text in SAMPLE_TEXTS:
            r = pipeline.classify(text)
            tox = r["toxicity"]["label"]
            intent = r["intent"]["label"]
            topic = r["topic"]["label"]
            risk = r["risk_level"]
            print(f"{text[:44]:<45} {tox:<12} {intent:<12} {topic:<12} {risk}")

    elif args.cmd == "analytics":
        pipeline = BehavioralPipeline()
        pipeline.train_all()
        stats = pipeline.analytics(SAMPLE_TEXTS)
        print("Analytics over sample corpus:")
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()
