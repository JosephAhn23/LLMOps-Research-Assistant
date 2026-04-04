"""
Multimodal RAG Pipeline.

Components:
  - CLIPImageIndex        — FAISS-backed image retrieval using CLIP embeddings
                            (text→image, image→image, fused multimodal queries)
  - VisionLanguageModel   — LLaVA-1.5 VQA + captioning, lazy-loaded
  - RAGImageGenerator     — RAG-enriched Stable Diffusion, lazy-loaded
  - MultimodalRAGPipeline — orchestrator combining all three

FastAPI route snippets at the bottom can be pasted into api/main.py.

Requirements: torch, transformers, diffusers, pillow, faiss-cpu
GPU recommended for LLaVA and Stable Diffusion inference.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MultimodalConfig:
    # Vision encoder
    clip_model: str = "openai/clip-vit-base-patch32"
    # VQA / captioning
    vlm_model: str = "llava-hf/llava-1.5-7b-hf"
    # Image generation
    diffusion_model: str = "stabilityai/stable-diffusion-2-1"
    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.25
    # Generation
    max_new_tokens: int = 256
    diffusion_steps: int = 25
    guidance_scale: float = 7.5
    image_size: int = 512
    # Output
    output_dir: str = "./outputs/multimodal"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# CLIP Image Index
# ---------------------------------------------------------------------------

class CLIPImageIndex:
    """
    FAISS-backed image index using CLIP embeddings.
    Supports text-to-image and image-to-image retrieval, plus fused queries.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor
        import faiss

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP: %s", model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.dim = 512  # CLIP ViT-B/32 embedding dim
        self.index = faiss.IndexFlatIP(self.dim)  # inner product = cosine on L2-normalised vecs
        self.metadata: List[Dict] = []

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def add_images(
        self,
        image_paths: List[Union[str, Path]],
        captions: Optional[List[str]] = None,
    ) -> None:
        """Index images from disk paths."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        embeddings = self._encode_images(images)
        self.index.add(embeddings)
        for i, path in enumerate(image_paths):
            self.metadata.append({
                "path": str(path),
                "caption": captions[i] if captions else "",
                "index": len(self.metadata),
            })
        logger.info("Indexed %d images. Total: %d", len(image_paths), self.index.ntotal)

    def add_pil_images(
        self,
        images: List[Image.Image],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Index PIL images directly (no disk path required)."""
        if not images:
            return
        embeddings = self._encode_images(images)
        self.index.add(embeddings)
        for i, _ in enumerate(images):
            meta = (metadata or [{}] * len(images))[i]
            self.metadata.append({"index": len(self.metadata), **meta})
        logger.info("Indexed %d PIL images (total: %d)", len(images), self.index.ntotal)

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top-k images matching a text query."""
        if self.index.ntotal == 0:
            return []
        emb = self._encode_text([query])
        scores, indices = self.index.search(emb, min(top_k, self.index.ntotal))
        return [
            {**self.metadata[idx], "score": float(scores[0][rank])}
            for rank, idx in enumerate(indices[0])
            if idx >= 0 and idx < len(self.metadata)
        ]

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Return top-k visually similar images."""
        if self.index.ntotal == 0:
            return []
        emb = self._encode_images([image])
        scores, indices = self.index.search(emb, min(top_k, self.index.ntotal))
        return [
            {**self.metadata[idx], "score": float(scores[0][rank])}
            for rank, idx in enumerate(indices[0])
            if idx >= 0 and idx < len(self.metadata)
        ]

    def search_multimodal(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict]:
        """
        Fuse text and image query embeddings (weighted sum) for multimodal retrieval.
        alpha=1.0 → text only, alpha=0.0 → image only.
        """
        if self.index.ntotal == 0:
            return []
        text_emb = self._encode_text([text])
        if image is not None:
            img_emb = self._encode_images([image])
            fused = alpha * text_emb + (1 - alpha) * img_emb
            fused = fused / (np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-8)
        else:
            fused = text_emb
        scores, indices = self.index.search(fused.astype("float32"), min(top_k, self.index.ntotal))
        return [
            {**self.metadata[idx], "score": float(scores[0][rank])}
            for rank, idx in enumerate(indices[0])
            if idx >= 0 and idx < len(self.metadata)
        ]

    def save(self, path: str) -> None:
        import faiss
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str) -> None:
        import faiss
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)


# ---------------------------------------------------------------------------
# Vision Language Model (LLaVA-1.5)
# ---------------------------------------------------------------------------

class VisionLanguageModel:
    """
    LLaVA-1.5 VQA and captioning. Lazy-loaded on first use.
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        logger.info("Loading VLM: %s (requires GPU for full performance)", self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str, max_new_tokens: int = 256) -> str:
        """Answer a question grounded in the provided image."""
        self._load()
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self._processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in decoded:
            return decoded.split("ASSISTANT:")[-1].strip()
        return decoded.strip()

    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        """Generate a detailed caption for an image."""
        return self.answer(image, "Describe this image in detail.")


# ---------------------------------------------------------------------------
# RAG Image Generator (Stable Diffusion + retrieved captions)
# ---------------------------------------------------------------------------

class RAGImageGenerator:
    """
    RAG-enriched Stable Diffusion image generation. Lazy-loaded on first use.

    Retrieves relevant captions from the CLIP index, builds an enriched
    prompt, and generates an image conditioned on retrieved context.
    Uses DPM-Solver++ for fast inference (25 steps vs default 50).
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        logger.info("Loading Stable Diffusion: %s", self.model_name)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )
        if self.device == "cpu":
            self._pipe.enable_attention_slicing()

    def generate(
        self,
        user_query: str,
        retrieved_captions: Optional[List[str]] = None,
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """
        Build a RAG-enriched prompt from retrieved captions + user query,
        then generate an image.
        """
        self._load()
        context = "; ".join((retrieved_captions or [])[:3])
        enriched_prompt = f"{user_query}. Style context: {context}" if context else user_query
        enriched_prompt = enriched_prompt[:400]  # SD prompt length limit
        logger.info("Generating image with prompt: %s...", enriched_prompt[:80])
        return self._pipe(
            prompt=enriched_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]


# ---------------------------------------------------------------------------
# Multimodal RAG Pipeline (orchestrator)
# ---------------------------------------------------------------------------

class MultimodalRAGPipeline:
    """
    Full multimodal pipeline:
      1. Index images with CLIP
      2. Retrieve relevant images for a text or image query
      3. VQA — answer questions grounded in retrieved images
      4. image_to_image_qa — answer questions about an uploaded image,
         enriched with retrieved similar images
      5. generate_image — RAG-grounded Stable Diffusion generation

    VLM and diffusion model are lazy-loaded on first use.
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self.clip_index = CLIPImageIndex(self.config.clip_model, device=self.config.device)
        self._vlm: Optional[VisionLanguageModel] = None
        self._generator: Optional[RAGImageGenerator] = None

    @property
    def vlm(self) -> VisionLanguageModel:
        if self._vlm is None:
            self._vlm = VisionLanguageModel(self.config.vlm_model, device=self.config.device)
        return self._vlm

    @property
    def generator(self) -> RAGImageGenerator:
        if self._generator is None:
            self._generator = RAGImageGenerator(
                self.config.diffusion_model, device=self.config.device
            )
        return self._generator

    def index(
        self,
        image_paths: List[str],
        captions: Optional[List[str]] = None,
    ) -> None:
        """Index images from disk paths."""
        self.clip_index.add_images(image_paths, captions)

    def visual_qa(self, query: str) -> Dict:
        """
        Answer a text query using retrieved images as grounding context.
        Filters retrieved images by similarity_threshold.
        """
        retrieved = self.clip_index.search_by_text(query, top_k=self.config.top_k)
        retrieved = [r for r in retrieved if r["score"] >= self.config.similarity_threshold]

        answers = []
        for item in retrieved[:3]:
            path = item.get("path")
            if not path or not Path(path).exists():
                continue
            img = Image.open(path).convert("RGB")
            ans = self.vlm.answer(img, query, max_new_tokens=self.config.max_new_tokens)
            answers.append({"source": path, "score": item["score"], "answer": ans})

        return {"query": query, "retrieved_count": len(retrieved), "answers": answers}

    def image_to_image_qa(self, image: Image.Image, question: str) -> Dict:
        """
        Answer a question about an uploaded image, enriched by retrieved similar images.
        """
        # Direct answer from the query image
        direct_answer = self.vlm.answer(image, question, max_new_tokens=self.config.max_new_tokens)

        # Retrieve similar images for additional context
        similar = self.clip_index.search_by_image(image, top_k=3)
        context_answers = []
        for item in similar[:2]:
            path = item.get("path")
            if not path or not Path(path).exists():
                continue
            ctx_img = Image.open(path).convert("RGB")
            ctx_ans = self.vlm.answer(
                ctx_img,
                f"Related context: {question}",
                max_new_tokens=self.config.max_new_tokens,
            )
            context_answers.append(ctx_ans)

        return {
            "question": question,
            "direct_answer": direct_answer,
            "context_answers": context_answers,
            "similar_images": [r.get("path", "") for r in similar],
        }

    def generate_image(self, query: str) -> Image.Image:
        """Retrieve relevant captions, then generate a RAG-grounded image."""
        retrieved = self.clip_index.search_by_text(query, top_k=3)
        captions = [r.get("caption", "") for r in retrieved if r.get("caption")]
        return self.generator.generate(
            query,
            retrieved_captions=captions,
            num_inference_steps=self.config.diffusion_steps,
            guidance_scale=self.config.guidance_scale,
            width=self.config.image_size,
            height=self.config.image_size,
        )

    def multimodal_query(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        k: int = 5,
    ) -> List[Dict]:
        """Fused text + image retrieval."""
        return self.clip_index.search_multimodal(text, image=image, top_k=k)


# ---------------------------------------------------------------------------
# FastAPI route snippets (paste into api/main.py)
# ---------------------------------------------------------------------------

FASTAPI_MULTIMODAL_ROUTES = '''
# --- Paste into api/main.py ---

import base64, io
from pathlib import Path
from fastapi import UploadFile, File, Form
from PIL import Image
from multimodal.multimodal_pipeline import MultimodalRAGPipeline, MultimodalConfig

_multimodal_pipeline: MultimodalRAGPipeline | None = None

def get_multimodal_pipeline() -> MultimodalRAGPipeline:
    global _multimodal_pipeline
    if _multimodal_pipeline is None:
        _multimodal_pipeline = MultimodalRAGPipeline()
    return _multimodal_pipeline


@app.get("/multimodal/search")
async def multimodal_search(query: str, k: int = 5):
    """Retrieve images matching a text query."""
    results = get_multimodal_pipeline().clip_index.search_by_text(query, top_k=k)
    return {"query": query, "results": results}


@app.post("/multimodal/vqa")
async def visual_qa(
    query: str = Form(...),
    image: UploadFile | None = File(None),
):
    """Answer a question grounded in retrieved images (or an uploaded image)."""
    pipeline = get_multimodal_pipeline()
    if image:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        return pipeline.image_to_image_qa(img, query)
    return pipeline.visual_qa(query)


@app.post("/multimodal/generate")
async def generate_image(query: str = Form(...)):
    """Generate a RAG-grounded image; returns base64-encoded PNG."""
    image = get_multimodal_pipeline().generate_image(query)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode()}


@app.post("/multimodal/index")
async def index_images(images: list[UploadFile] = File(...)):
    """Index uploaded images into the CLIP vector store."""
    pipeline = get_multimodal_pipeline()
    Path(pipeline.config.output_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for f in images:
        dest = Path(pipeline.config.output_dir) / f.filename
        dest.write_bytes(await f.read())
        paths.append(str(dest))
    pipeline.index(paths)
    return {"indexed": len(paths)}
'''


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Multimodal RAG Pipeline demo (CPU mode — CLIP only, no GPU required)")

    config = MultimodalConfig(device="cpu")
    pipeline = MultimodalRAGPipeline(config)

    # Synthetic demo images (solid colours)
    demo_images = [
        Image.new("RGB", (224, 224), color=(int(i * 40) % 255, 100, 200 - i * 20))
        for i in range(5)
    ]
    captions = [
        "a neural network architecture diagram",
        "FAISS vector similarity search visualization",
        "transformer attention heatmap",
        "RAG pipeline flowchart",
        "LLM fine-tuning loss curves",
    ]
    pipeline.clip_index.add_pil_images(demo_images, [{"caption": c} for c in captions])
    logger.info("Indexed %d demo images", len(demo_images))

    results = pipeline.clip_index.search_by_text("neural network training", top_k=3)
    logger.info("Text→image search results:")
    for r in results:
        logger.info("  [%.3f] %s", r["score"], r.get("caption", ""))

    results2 = pipeline.multimodal_query("vector search", image=demo_images[0], k=3)
    logger.info("Multimodal (fused) search results:")
    for r in results2:
        logger.info("  [%.3f] %s", r["score"], r.get("caption", ""))

    logger.info("Demo complete. VLM and Stable Diffusion load lazily on first use (GPU recommended).")


if __name__ == "__main__":
    demo()
