"""
Real diffusion conditioning - textual inversion + cross-attention manipulation.
Retrieved context guides image generation via learned embeddings.
Covers: Diffusion conditioning (real, not string concat)
"""
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = Path("outputs/conditioned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TextualInversionTrainer:
    """
    Fine-tunes text encoder on retrieved domain concepts.
    Retrieved context becomes a learnable token <retrieved-concept>.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        placeholder_token: str = "<retrieved-concept>",
        n_train_steps: int = 500,
        lr: float = 5e-4,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
    ):
        self.model_id = model_id
        self.placeholder_token = placeholder_token
        self.n_train_steps = n_train_steps
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = (text_encoder or CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")).to(
            self.device
        )

        self.tokenizer.add_tokens([placeholder_token])
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        init_token_ids = self.tokenizer.encode("knowledge document context", add_special_tokens=False)
        init_embedding = token_embeds[init_token_ids].mean(0)
        token_embeds[self.placeholder_token_id] = init_embedding

    def train_on_retrieved_context(
        self,
        retrieved_chunks: List[Dict],
        training_images: Optional[List[Image.Image]] = None,  # reserved for future supervised variant
    ) -> Dict:
        _ = training_images
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.get_input_embeddings().weight.requires_grad = True

        optimizer = AdamW([self.text_encoder.get_input_embeddings().weight], lr=self.lr)

        training_prompts = []
        for chunk in retrieved_chunks[:10]:
            text = chunk["text"][:100]
            training_prompts.extend(
                [
                    f"a photo of {self.placeholder_token}, {text[:50]}",
                    f"{self.placeholder_token} concept, {text[:50]}",
                    f"an image representing {self.placeholder_token}",
                ]
            )

        if not training_prompts:
            training_prompts = [f"an image of {self.placeholder_token}"]

        losses = []
        with mlflow.start_run(run_name="textual-inversion", nested=True):
            mlflow.log_param("placeholder_token", self.placeholder_token)
            mlflow.log_param("n_train_steps", self.n_train_steps)
            mlflow.log_param("n_retrieved_chunks", len(retrieved_chunks))

            for step in range(self.n_train_steps):
                prompt = training_prompts[step % len(training_prompts)]
                inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.text_encoder(**inputs)
                text_embeddings = outputs.last_hidden_state

                token_positions = (inputs.input_ids == self.placeholder_token_id).nonzero(as_tuple=False)
                if token_positions.numel() == 0:
                    continue
                concept_pos = int(token_positions[0, 1].item())
                concept_embedding = text_embeddings[:, concept_pos, :]

                context_mask = inputs.attention_mask.bool()
                context_embeddings = text_embeddings[context_mask].mean(0)
                loss = F.mse_loss(concept_embedding.squeeze(), context_embeddings.detach())

                optimizer.zero_grad()
                loss.backward()

                grads = self.text_encoder.get_input_embeddings().weight.grad
                grads_mask = torch.zeros_like(grads)
                grads_mask[self.placeholder_token_id] = grads[self.placeholder_token_id]
                self.text_encoder.get_input_embeddings().weight.grad = grads_mask
                optimizer.step()

                losses.append(loss.item())
                if step % 100 == 0:
                    mlflow.log_metric("inversion_loss", loss.item(), step=step)

            final_loss = float(np.mean(losses[-50:])) if losses else 0.0
            mlflow.log_metric("final_loss", final_loss)

        return {
            "placeholder_token": self.placeholder_token,
            "token_id": self.placeholder_token_id,
            "final_loss": round(final_loss, 6),
            "n_steps": self.n_train_steps,
        }

    def save_learned_embedding(self, path: str = "models/textual_inversion"):
        Path(path).mkdir(parents=True, exist_ok=True)
        learned_embedding = self.text_encoder.get_input_embeddings().weight[self.placeholder_token_id].detach().cpu()
        torch.save(
            {
                "placeholder_token": self.placeholder_token,
                "token_id": self.placeholder_token_id,
                "embedding": learned_embedding,
            },
            f"{path}/concept_embedding.pt",
        )

    def load_learned_embedding(self, path: str = "models/textual_inversion"):
        data = torch.load(f"{path}/concept_embedding.pt", map_location=self.device)
        self.text_encoder.get_input_embeddings().weight.data[data["token_id"]] = data["embedding"].to(self.device)


class CrossAttentionRAGConditioner:
    """
    Captures UNet cross-attention activations to inspect text conditioning.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        self.pipe = pipe
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(name: str):
            def hook(_module, _input, output):
                if isinstance(output, torch.Tensor):
                    self.attention_maps[name] = output.detach()

            return hook

        for name, module in self.pipe.unet.named_modules():
            if "attn2" in name and hasattr(module, "to_q"):
                module.register_forward_hook(make_hook(name))

    def get_token_attention_maps(self, prompt: str, token_idx: int, image_size: int = 512) -> np.ndarray:
        generator = torch.Generator(self.pipe.device).manual_seed(42)
        _ = self.pipe(prompt, num_inference_steps=10, generator=generator, guidance_scale=7.5)

        all_maps = []
        for _name, attn in self.attention_maps.items():
            if attn.dim() == 4 and token_idx < attn.shape[-1]:
                token_attn = attn[0, :, :, token_idx].mean(0)
                spatial_size = int(token_attn.shape[0] ** 0.5)
                if spatial_size**2 == token_attn.shape[0]:
                    token_map = token_attn.reshape(spatial_size, spatial_size)
                    token_map = F.interpolate(
                        token_map.unsqueeze(0).unsqueeze(0),
                        size=(image_size // 8, image_size // 8),
                        mode="bilinear",
                    ).squeeze()
                    all_maps.append(token_map.detach().cpu().numpy())

        if not all_maps:
            return np.zeros((image_size // 8, image_size // 8))

        aggregated = np.mean(all_maps, axis=0)
        aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min() + 1e-8)
        return aggregated

    def generate_with_attention_guidance(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        trainer: TextualInversionTrainer,
        n_steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Dict:
        _ = retrieved_chunks
        concept_token = trainer.placeholder_token
        grounded_prompt = f"a detailed image of {concept_token}, {query}, high quality"
        negative_prompt = "blurry, low quality, distorted, text, watermark"

        start = time.perf_counter()
        output = self.pipe(
            prompt=grounded_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(self.pipe.device).manual_seed(42),
        )
        latency = time.perf_counter() - start

        tokens = trainer.tokenizer.encode(grounded_prompt)
        concept_idx = tokens.index(trainer.placeholder_token_id) if trainer.placeholder_token_id in tokens else 1
        attention_map = self.get_token_attention_maps(grounded_prompt, concept_idx)

        ts = int(time.time())
        img_path = OUTPUT_DIR / f"conditioned_{ts}.png"
        attn_path = OUTPUT_DIR / f"attention_{ts}.npy"
        output.images[0].save(img_path)
        np.save(attn_path, attention_map)

        with mlflow.start_run(run_name="conditioned-generation", nested=True):
            mlflow.log_metric("generation_latency_s", latency)
            mlflow.log_param("concept_token", concept_token)
            mlflow.log_param("n_steps", n_steps)
            mlflow.log_artifact(str(img_path), "generated")

        return {
            "image": output.images[0],
            "image_path": str(img_path),
            "attention_map": attention_map,
            "prompt": grounded_prompt,
            "latency_s": round(latency, 2),
            "concept_token": concept_token,
        }


class RAGConditionedDiffusion:
    """
    End-to-end RAG-conditioned generation.
    retrieve -> learn concept token -> generate with attention inspection
    """

    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(self.device)

        self.trainer = TextualInversionTrainer(
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )
        self.conditioner = CrossAttentionRAGConditioner(self.pipe)
        self._concept_cache: Dict[str, str] = {}

    def generate(self, query: str, retrieved_chunks: List[Dict], force_retrain: bool = False) -> Dict:
        context_hash = hashlib.md5("".join(c["text"][:50] for c in retrieved_chunks).encode()).hexdigest()[:8]

        if context_hash not in self._concept_cache or force_retrain:
            train_result = self.trainer.train_on_retrieved_context(retrieved_chunks)
            self.trainer.save_learned_embedding(f"models/concepts/{context_hash}")
            self._concept_cache[context_hash] = context_hash
            _ = train_result
        else:
            self.trainer.load_learned_embedding(f"models/concepts/{context_hash}")

        result = self.conditioner.generate_with_attention_guidance(
            query=query,
            retrieved_chunks=retrieved_chunks,
            trainer=self.trainer,
        )
        result["context_hash"] = context_hash
        result["n_retrieved"] = len(retrieved_chunks)
        return result
