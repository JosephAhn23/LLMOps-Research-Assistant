"""
Diffusion model integration - text-to-image generation with RAG grounding.
Covers: Diffusion models, multimodal generation, post-training
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import torch
from PIL import Image
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs/generated_images")


class RAGGroundedDiffusion:
    """
    RAG-grounded image generation - retrieves relevant context,
    conditions diffusion model on retrieved text.
    Covers: Diffusion models + multimodal RAG
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipe = None
        self.img2img_pipe = None
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_pipeline(self, low_vram: bool = False):
        """Load SD pipeline with optimized scheduler."""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        # DPM-Solver++ - 20 steps vs default 50, similar quality
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )

        if low_vram:
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

        self.pipe = self.pipe.to(self.device)

        # img2img for RAG-conditioned generation
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)
        self.img2img_pipe = self.img2img_pipe.to(self.device)

    def generate_from_rag_context(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        n_images: int = 4,
        guidance_scale: float = 7.5,
        n_steps: int = 20,
        negative_prompt: str = "blurry, low quality, distorted, ugly",
    ) -> Dict:
        """
        Generate images conditioned on RAG-retrieved context.
        Synthesizes prompt from retrieved chunks + original query.
        """
        if self.pipe is None:
            self.load_pipeline()

        context_summary = " ".join([c["text"][:100] for c in retrieved_chunks[:3]])
        grounded_prompt = f"{query}. Context: {context_summary}"

        with mlflow.start_run(run_name="diffusion-rag", nested=True):
            mlflow.log_param("query", query[:200])
            mlflow.log_param("n_steps", n_steps)
            mlflow.log_param("guidance_scale", guidance_scale)
            mlflow.log_param("n_retrieved_chunks", len(retrieved_chunks))

            start = time.perf_counter()
            output = self.pipe(
                prompt=grounded_prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=n_images,
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(self.device).manual_seed(42),
            )
            latency = time.perf_counter() - start

            mlflow.log_metric("generation_latency_s", round(latency, 2))
            mlflow.log_metric("images_generated", len(output.images))

            saved_paths = []
            for i, image in enumerate(output.images):
                path = OUTPUT_DIR / f"rag_gen_{int(time.time())}_{i}.png"
                image.save(path)
                saved_paths.append(str(path))
                mlflow.log_artifact(str(path), "generated_images")

        return {
            "prompt": grounded_prompt,
            "images": output.images,
            "saved_paths": saved_paths,
            "latency_s": round(latency, 2),
            "n_steps": n_steps,
        }

    def benchmark_scheduler_comparison(self) -> Dict:
        """
        Compare schedulers: DDPM vs DPM-Solver++ vs PNDM.
        Covers: Diffusion model latency reduction
        """
        from diffusers import DDPMScheduler, PNDMScheduler

        if self.pipe is None:
            self.load_pipeline()

        schedulers = {
            "ddpm_50steps": (DDPMScheduler.from_config(self.pipe.scheduler.config), 50),
            "pndm_50steps": (PNDMScheduler.from_config(self.pipe.scheduler.config), 50),
            "dpm_solver_20steps": (
                DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                ),
                20,
            ),
            "dpm_solver_10steps": (
                DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                ),
                10,
            ),
        }

        results = {}
        test_prompt = "a photorealistic mountain landscape with snow peaks"

        with mlflow.start_run(run_name="scheduler-benchmark"):
            for name, (scheduler, steps) in schedulers.items():
                self.pipe.scheduler = scheduler
                start = time.perf_counter()
                output = self.pipe(
                    test_prompt,
                    num_inference_steps=steps,
                    generator=torch.Generator(self.device).manual_seed(42),
                )
                latency = time.perf_counter() - start

                path = OUTPUT_DIR / f"scheduler_{name}.png"
                output.images[0].save(path)

                results[name] = {
                    "latency_s": round(latency, 2),
                    "steps": steps,
                    "ms_per_step": round(latency * 1000 / steps, 1),
                }
                mlflow.log_metric(f"{name}_latency_s", latency)
                print(f"  {name}: {latency:.2f}s ({steps} steps)")

        return results


class MultimodalRAGPipeline:
    """
    Multimodal RAG - accepts text + image queries.
    Covers: Multimodal LLMs, text+image RAG
    """

    def __init__(self):
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image_query(self, image_path: str) -> np.ndarray:
        """Encode image as embedding for multimodal retrieval."""
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        embedding = image_features.numpy()
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding

    def multimodal_query(
        self,
        text_query: Optional[str] = None,
        image_path: Optional[str] = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Fuse text + image embeddings for multimodal retrieval.
        alpha controls text vs image weighting.
        """
        embeddings = []

        if text_query:
            from ingestion.pipeline import EmbeddingModel

            text_emb = EmbeddingModel().embed([text_query])
            embeddings.append(("text", text_emb, alpha))

        if image_path:
            img_emb = self.encode_image_query(image_path)
            embeddings.append(("image", img_emb, 1 - alpha))

        if not embeddings:
            raise ValueError("Must provide text_query or image_path")

        if len(embeddings) == 1:
            return embeddings[0][1]

        fused = sum(w * emb for _, emb, w in embeddings)
        fused = fused / np.linalg.norm(fused, axis=1, keepdims=True)
        return fused
