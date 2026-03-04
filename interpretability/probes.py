"""
Interpretability & Probing Suite
=================================
Attention visualization, linear probes on hidden states, activation patching,
and Centered Kernel Alignment (CKA) for representation similarity.

All components run on CPU with any small HuggingFace model.
No GPU required; tested with bert-base-uncased and gpt2.

Components:
  1. AttentionVisualizer   — extract + render attention patterns as ASCII/HTML
  2. LinearProbe           — train a logistic probe on hidden-state activations
  3. ActivationPatcher     — causal intervention: patch activations from a
                             "clean" run into a "corrupted" run to isolate
                             which positions/layers cause a behaviour
  4. CKAAnalyzer           — Centered Kernel Alignment between layer pairs
                             or between two models
  5. HiddenStateExtractor  — hook-based extractor for any HF model

Usage:
    extractor = HiddenStateExtractor("bert-base-uncased")
    states = extractor.extract(["The cat sat", "The dog ran"])

    probe = LinearProbe(input_dim=768, num_classes=2)
    probe.train(states["layer_6"], labels)
    acc = probe.evaluate(states["layer_6"], labels)

    cka = CKAAnalyzer()
    similarity = cka.linear_cka(states["layer_3"], states["layer_11"])
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hidden-state extractor (forward hooks)
# ──────────────────────────────────────────────────────────────────────────────

class HiddenStateExtractor:
    """
    Extract hidden states and attention weights from any HuggingFace model
    using forward hooks. No model surgery required.

    Returns:
        {
          "layer_0": Tensor[batch, seq, hidden],
          "layer_1": ...,
          "attention_layer_0": Tensor[batch, heads, seq, seq],
          ...
        }
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device).eval()
        self.model_name = model_name

    @torch.no_grad()
    def extract(
        self,
        texts: List[str],
        max_length: int = 128,
        pooling: str = "mean",   # mean | cls | none
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            texts:    list of input strings
            pooling:  how to reduce the seq dimension
                      "mean" → mean over non-padding tokens
                      "cls"  → first token (BERT [CLS])
                      "none" → keep full [batch, seq, hidden]

        Returns:
            dict mapping "layer_N" and "attention_layer_N" to tensors
        """
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        outputs = self.model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, S, 1]

        result: Dict[str, torch.Tensor] = {}

        # Hidden states: tuple of (num_layers+1) tensors [B, S, H]
        for i, hs in enumerate(outputs.hidden_states):
            if pooling == "cls":
                result[f"layer_{i}"] = hs[:, 0, :].cpu()
            elif pooling == "mean":
                result[f"layer_{i}"] = (hs * mask).sum(1) / mask.sum(1)
                result[f"layer_{i}"] = result[f"layer_{i}"].cpu()
            else:
                result[f"layer_{i}"] = hs.cpu()

        # Attention weights: tuple of num_layers tensors [B, H, S, S]
        if outputs.attentions:
            for i, attn in enumerate(outputs.attentions):
                result[f"attention_layer_{i}"] = attn.cpu()

        return result

    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def hidden_size(self) -> int:
        return self.model.config.hidden_size


# ──────────────────────────────────────────────────────────────────────────────
# 2. Attention visualizer
# ──────────────────────────────────────────────────────────────────────────────

class AttentionVisualizer:
    """
    Visualize attention patterns as ASCII heatmaps or HTML.

    Supports:
      - Single head visualization
      - Mean-head aggregation
      - Head importance ranking (by entropy — low entropy = more focused)
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        self.extractor = HiddenStateExtractor(model_name, device)

    def get_attention(
        self, text: str, layer: int = 0
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Returns:
            attn:   [num_heads, seq, seq] attention weights for one text
            tokens: list of token strings
        """
        enc = self.extractor.tokenizer(text, return_tensors="pt").to(self.extractor.device)
        tokens = self.extractor.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        with torch.no_grad():
            outputs = self.extractor.model(**enc)

        attn = outputs.attentions[layer][0].cpu()  # [heads, seq, seq]
        return attn, tokens

    def ascii_heatmap(
        self,
        text: str,
        layer: int = 0,
        head: Optional[int] = None,
        width: int = 60,
    ) -> str:
        """Render an attention matrix as an ASCII heatmap."""
        attn, tokens = self.get_attention(text, layer)

        if head is not None:
            matrix = attn[head]  # [seq, seq]
        else:
            matrix = attn.mean(0)  # mean over heads

        seq = len(tokens)
        chars = " ░▒▓█"
        lines = [f"Layer {layer}" + (f" head {head}" if head is not None else " (mean)")]
        lines.append("      " + "".join(f"{t[:4]:>5}" for t in tokens[:seq]))

        for i, row_tok in enumerate(tokens[:seq]):
            row = matrix[i, :seq]
            row_norm = row / row.max().clamp(min=1e-8)
            bar = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)]
                          for v in row_norm.tolist())
            lines.append(f"{row_tok[:5]:>5} {bar}")

        return "\n".join(lines)

    def head_entropy(self, text: str, layer: int = 0) -> List[Tuple[int, float]]:
        """
        Rank attention heads by entropy.
        Low entropy → focused attention (potentially more interpretable).
        """
        attn, _ = self.get_attention(text, layer)
        results = []
        for h in range(attn.shape[0]):
            mat = attn[h]  # [seq, seq]
            # Mean entropy over query positions
            ent = -(mat * (mat + 1e-9).log()).sum(-1).mean().item()
            results.append((h, round(ent, 4)))
        return sorted(results, key=lambda x: x[1])  # ascending = most focused first

    def html_heatmap(self, text: str, layer: int = 0, head: int = 0) -> str:
        """Generate an HTML table with colour-coded attention weights."""
        attn, tokens = self.get_attention(text, layer)
        matrix = attn[head].tolist()
        seq = len(tokens)

        rows = []
        rows.append("<table style='font-family:monospace;font-size:11px;border-collapse:collapse'>")
        rows.append("<tr><th></th>" + "".join(f"<th>{t}</th>" for t in tokens[:seq]) + "</tr>")
        for i, tok in enumerate(tokens[:seq]):
            cells = [f"<td><b>{tok}</b></td>"]
            for j in range(seq):
                v = matrix[i][j]
                alpha = int(v * 255)
                cells.append(
                    f"<td style='background:rgba(0,100,200,{v:.3f});padding:2px'>{v:.2f}</td>"
                )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        rows.append("</table>")
        return "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Linear probe
# ──────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """
    Logistic regression probe on frozen hidden states.

    Used to test whether a concept (e.g. sentiment, POS tag, syntactic role)
    is linearly decodable from a specific layer's representations.

    High accuracy → the concept is encoded in that layer.
    Accuracy increasing with depth → concept is built up gradually.
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def train_probe(
        self,
        X: torch.Tensor,       # [N, hidden_dim]
        y: torch.Tensor,       # [N] int labels
        epochs: int = 100,
        lr: float = 1e-2,
        l2: float = 1e-4,
        verbose: bool = False,
    ) -> List[float]:
        """Train the probe and return per-epoch loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.train()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if verbose and (epoch + 1) % 20 == 0:
                logger.info("Probe epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss.item())
        return losses

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        self.eval()
        logits = self(X)
        preds = logits.argmax(-1)
        acc = (preds == y).float().mean().item()
        # Per-class accuracy
        per_class = {}
        for c in y.unique().tolist():
            mask = y == c
            per_class[int(c)] = round((preds[mask] == y[mask]).float().mean().item(), 4)
        return {
            "accuracy": round(acc, 4),
            "per_class_accuracy": per_class,
            "num_samples": len(y),
        }


class LayerwiseProbeExperiment:
    """
    Train a linear probe at every layer and plot accuracy vs depth.
    Reveals where in the network a concept becomes linearly decodable.
    """

    def __init__(self, extractor: HiddenStateExtractor):
        self.extractor = extractor

    def run(
        self,
        texts: List[str],
        labels: List[int],
        num_classes: int = 2,
        epochs: int = 100,
        test_split: float = 0.2,
    ) -> Dict[str, Dict]:
        """
        Returns:
            {
              "layer_0": {"train_acc": 0.72, "test_acc": 0.68},
              "layer_1": ...,
              ...
            }
        """
        states = self.extractor.extract(texts, pooling="mean")
        y = torch.tensor(labels, dtype=torch.long)

        n = len(texts)
        n_test = max(1, int(n * test_split))
        idx = torch.randperm(n)
        train_idx, test_idx = idx[n_test:], idx[:n_test]

        results = {}
        num_layers = self.extractor.num_layers()

        for layer_i in range(num_layers + 1):
            key = f"layer_{layer_i}"
            if key not in states:
                continue

            X = states[key]  # [N, H]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            probe = LinearProbe(input_dim=X.shape[-1], num_classes=num_classes)
            probe.train_probe(X_train, y_train, epochs=epochs)

            train_eval = probe.evaluate(X_train, y_train)
            test_eval = probe.evaluate(X_test, y_test)

            results[key] = {
                "train_acc": train_eval["accuracy"],
                "test_acc": test_eval["accuracy"],
            }
            logger.info(
                "Layer %2d: train_acc=%.3f  test_acc=%.3f",
                layer_i, train_eval["accuracy"], test_eval["accuracy"],
            )

        return results

    def ascii_plot(self, results: Dict[str, Dict], metric: str = "test_acc") -> str:
        """Render layer-wise accuracy as an ASCII bar chart."""
        layers = sorted(results.keys(), key=lambda k: int(k.split("_")[1]))
        lines = [f"Layer-wise probe accuracy ({metric})", ""]
        for layer in layers:
            val = results[layer][metric]
            bar_len = int(val * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"{layer:>10}: {bar} {val:.3f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Activation patcher (causal tracing)
# ──────────────────────────────────────────────────────────────────────────────

class ActivationPatcher:
    """
    Causal activation patching (Meng et al., 2022 — ROME).

    Protocol:
      1. Run model on "clean" input → cache activations at every layer/position
      2. Run model on "corrupted" input (e.g. with a key token replaced)
      3. For each (layer, position), patch the corrupted run with the clean
         activation and measure the change in output probability
      4. High recovery → that (layer, position) is causally important

    This implementation patches hidden states (residual stream).
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device).eval()

    @torch.no_grad()
    def _run(self, text: str) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Run model, return logits and all hidden states."""
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        return out.logits[0], out.hidden_states  # logits: [seq, vocab]

    def patch_experiment(
        self,
        clean_text: str,
        corrupted_text: str,
        target_token: str,
        patch_positions: Optional[List[int]] = None,
    ) -> Dict:
        """
        Measure how much patching each layer's activations from the clean run
        into the corrupted run recovers the probability of target_token.

        Args:
            clean_text:      the "correct" input
            corrupted_text:  the "broken" input (e.g. subject replaced)
            target_token:    the token whose probability we track
            patch_positions: which token positions to patch (None = all)

        Returns:
            {
              "clean_prob":     float,
              "corrupted_prob": float,
              "layer_effects":  {layer_idx: recovered_prob},
            }
        """
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)
        if not target_id:
            raise ValueError(f"Token {target_token!r} not in vocabulary")
        target_id = target_id[0]

        clean_logits, clean_states = self._run(clean_text)
        corr_logits, corr_states = self._run(corrupted_text)

        def last_token_prob(logits: torch.Tensor) -> float:
            return F.softmax(logits[-1], dim=-1)[target_id].item()

        clean_prob = last_token_prob(clean_logits)
        corr_prob = last_token_prob(corr_logits)

        layer_effects: Dict[int, float] = {}
        num_layers = len(clean_states)

        enc_corr = self.tokenizer(corrupted_text, return_tensors="pt").to(self.device)
        seq_len = enc_corr["input_ids"].shape[1]
        positions = patch_positions or list(range(seq_len))

        for layer_idx in range(num_layers):
            # Patch clean activation into corrupted run at this layer
            hooks = []
            clean_act = clean_states[layer_idx]  # [1, seq, hidden]

            def make_hook(clean, positions):
                def hook_fn(module, input, output):
                    # output may be a tuple; hidden state is first element
                    hs = output[0] if isinstance(output, tuple) else output
                    patched = hs.clone()
                    for pos in positions:
                        if pos < clean.shape[1] and pos < hs.shape[1]:
                            patched[0, pos, :] = clean[0, pos, :]
                    if isinstance(output, tuple):
                        return (patched,) + output[1:]
                    return patched
                return hook_fn

            # Register hook on the layer_idx-th transformer block
            try:
                layers_module = (
                    self.model.transformer.h  # GPT-2
                    if hasattr(self.model, "transformer")
                    else self.model.model.layers  # LLaMA-style
                )
                if layer_idx < len(layers_module):
                    h = layers_module[layer_idx].register_forward_hook(
                        make_hook(clean_act, positions)
                    )
                    hooks.append(h)
            except AttributeError:
                pass

            with torch.no_grad():
                out = self.model(**enc_corr)
            patched_prob = last_token_prob(out.logits[0])
            layer_effects[layer_idx] = round(patched_prob, 6)

            for h in hooks:
                h.remove()

        return {
            "clean_prob": round(clean_prob, 6),
            "corrupted_prob": round(corr_prob, 6),
            "layer_effects": layer_effects,
            "target_token": target_token,
        }

    def ascii_causal_trace(self, results: Dict) -> str:
        """Render causal trace as ASCII bar chart."""
        clean = results["clean_prob"]
        corr = results["corrupted_prob"]
        lines = [
            f"Causal trace for token: {results['target_token']!r}",
            f"Clean prob:     {clean:.4f}",
            f"Corrupted prob: {corr:.4f}",
            "",
            "Layer-wise recovery (patching clean → corrupted):",
        ]
        for layer, prob in sorted(results["layer_effects"].items()):
            recovery = (prob - corr) / max(clean - corr, 1e-8)
            bar_len = max(0, int(recovery * 30))
            bar = "█" * bar_len
            lines.append(f"  Layer {layer:2d}: {bar:<30} {prob:.4f} (recovery={recovery:.2f})")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 5. CKA (Centered Kernel Alignment)
# ──────────────────────────────────────────────────────────────────────────────

class CKAAnalyzer:
    """
    Centered Kernel Alignment (Kornblith et al., 2019).

    Measures representational similarity between two sets of activations,
    invariant to orthogonal transformations and isotropic scaling.

    CKA = 1.0 → representations are identical (up to rotation/scale)
    CKA = 0.0 → representations are completely dissimilar

    Use cases:
      - Compare layers within a model (how similar are layer 3 and layer 11?)
      - Compare the same layer across two models (BERT vs RoBERTa)
      - Track representation drift during fine-tuning
    """

    @staticmethod
    def _center(K: torch.Tensor) -> torch.Tensor:
        """Double-center a kernel matrix."""
        n = K.shape[0]
        H = torch.eye(n) - torch.ones(n, n) / n
        return H @ K @ H

    def linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Linear CKA using dot-product kernels.

        Args:
            X: [n_samples, dim_x]
            Y: [n_samples, dim_y]

        Returns:
            CKA similarity in [0, 1]
        """
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

        # HSIC with linear kernels
        def hsic(A, B):
            n = A.shape[0]
            K = A @ A.T
            L = B @ B.T
            KH = self._center(K)
            LH = self._center(L)
            return (KH * LH).sum() / (n - 1) ** 2

        hsic_xy = hsic(X, Y)
        hsic_xx = hsic(X, X)
        hsic_yy = hsic(Y, Y)

        denom = (hsic_xx * hsic_yy).sqrt()
        if denom < 1e-10:
            return 0.0
        return (hsic_xy / denom).item()

    def rbf_cka(self, X: torch.Tensor, Y: torch.Tensor, sigma: Optional[float] = None) -> float:
        """
        RBF kernel CKA — more sensitive to non-linear similarity.
        sigma defaults to median pairwise distance heuristic.
        """
        def rbf_kernel(A, s):
            sq_dists = torch.cdist(A, A) ** 2
            if s is None:
                s = sq_dists.median().sqrt().item()
            return torch.exp(-sq_dists / (2 * s ** 2))

        Kx = self._center(rbf_kernel(X, sigma))
        Ky = self._center(rbf_kernel(Y, sigma))

        hsic_xy = (Kx * Ky).sum()
        denom = ((Kx * Kx).sum() * (Ky * Ky).sum()).sqrt()
        if denom < 1e-10:
            return 0.0
        return (hsic_xy / denom).item()

    def layer_similarity_matrix(
        self,
        states: Dict[str, torch.Tensor],
        kernel: str = "linear",
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Compute pairwise CKA between all layers.

        Args:
            states: {"layer_0": Tensor[N, H], "layer_1": ..., ...}
            kernel: "linear" | "rbf"

        Returns:
            similarity_matrix: [num_layers, num_layers]
            layer_names:       list of layer keys
        """
        layer_keys = sorted(
            [k for k in states if k.startswith("layer_")],
            key=lambda k: int(k.split("_")[1]),
        )
        n = len(layer_keys)
        sim = torch.zeros(n, n)

        fn = self.linear_cka if kernel == "linear" else self.rbf_cka

        for i, ki in enumerate(layer_keys):
            for j, kj in enumerate(layer_keys):
                if i <= j:
                    val = fn(states[ki].float(), states[kj].float())
                    sim[i, j] = val
                    sim[j, i] = val

        return sim, layer_keys

    def ascii_similarity_matrix(
        self,
        sim: torch.Tensor,
        layer_names: List[str],
    ) -> str:
        """Render the CKA similarity matrix as ASCII."""
        n = len(layer_names)
        short = [f"L{k.split('_')[1]}" for k in layer_names]
        header = "     " + "".join(f"{s:>5}" for s in short)
        lines = [header]
        chars = " ░▒▓█"
        for i, name in enumerate(short):
            row = f"{name:>4} "
            for j in range(n):
                v = sim[i, j].item()
                c = chars[min(int(v * (len(chars) - 1)), len(chars) - 1)]
                row += f"  {c}{c} "
            lines.append(row)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Interpretability & probing suite")
    sub = parser.add_subparsers(dest="cmd")

    attn_p = sub.add_parser("attention", help="Visualize attention patterns")
    attn_p.add_argument("--model", default="bert-base-uncased")
    attn_p.add_argument("--text", default="The cat sat on the mat")
    attn_p.add_argument("--layer", type=int, default=0)

    probe_p = sub.add_parser("probe", help="Layer-wise linear probe experiment")
    probe_p.add_argument("--model", default="bert-base-uncased")

    patch_p = sub.add_parser("patch", help="Activation patching / causal tracing")
    patch_p.add_argument("--model", default="gpt2")
    patch_p.add_argument("--clean", default="The Eiffel Tower is located in Paris")
    patch_p.add_argument("--corrupted", default="The Eiffel Tower is located in Berlin")
    patch_p.add_argument("--target", default=" Paris")

    cka_p = sub.add_parser("cka", help="CKA layer similarity matrix")
    cka_p.add_argument("--model", default="bert-base-uncased")

    args = parser.parse_args()

    SENTIMENT_TEXTS = [
        "I love this movie, it was fantastic!",
        "This film was absolutely wonderful.",
        "Great acting and beautiful cinematography.",
        "I hated this movie, it was terrible.",
        "Worst film I have ever seen.",
        "Boring, slow, and poorly written.",
        "An okay film, nothing special.",
        "Pretty average, not great not bad.",
    ]
    SENTIMENT_LABELS = [1, 1, 1, 0, 0, 0, 1, 0]  # 1=positive, 0=negative

    if args.cmd == "attention":
        print(f"Attention visualization: {args.model}")
        viz = AttentionVisualizer(args.model)
        print(viz.ascii_heatmap(args.text, layer=args.layer))
        print("\nHead entropy (most focused first):")
        for head, ent in viz.head_entropy(args.text, layer=args.layer)[:5]:
            print(f"  Head {head}: entropy={ent:.4f}")

    elif args.cmd == "probe":
        print(f"Layer-wise probe experiment: {args.model}")
        extractor = HiddenStateExtractor(args.model)
        exp = LayerwiseProbeExperiment(extractor)
        results = exp.run(SENTIMENT_TEXTS, SENTIMENT_LABELS, num_classes=2, epochs=200)
        print(exp.ascii_plot(results))

    elif args.cmd == "patch":
        print(f"Activation patching: {args.model}")
        patcher = ActivationPatcher(args.model)
        results = patcher.patch_experiment(args.clean, args.corrupted, args.target)
        print(patcher.ascii_causal_trace(results))

    elif args.cmd == "cka":
        print(f"CKA layer similarity: {args.model}")
        extractor = HiddenStateExtractor(args.model)
        texts = SENTIMENT_TEXTS
        states = extractor.extract(texts, pooling="mean")
        cka = CKAAnalyzer()
        sim, names = cka.layer_similarity_matrix(states)
        print(cka.ascii_similarity_matrix(sim, names))
        print(f"\nMost similar pair: ", end="")
        max_val, max_i, max_j = 0.0, 0, 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if sim[i, j].item() > max_val:
                    max_val, max_i, max_j = sim[i, j].item(), i, j
        print(f"{names[max_i]} ↔ {names[max_j]}: CKA={max_val:.4f}")

    else:
        parser.print_help()
