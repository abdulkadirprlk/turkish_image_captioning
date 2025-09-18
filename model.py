from typing import Any, Dict, List, Iterable, Optional
import os

import torch
import torch.nn as nn
import clip
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from PIL import Image as PILImage


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, prefix_tokens: int = 8, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim * prefix_tokens)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(x.size(0), self.prefix_tokens, -1)
        x = self.ln(x)
        return self.dropout(x)


class CLIPmT5Pipeline(nn.Module):
    """
    Unified train/inference pipeline used by both notebook training and Streamlit inference.
    - Forward(images, src_texts, tgt_texts) for training (teacher forcing)
    - generate(image_paths=..., images=...) for inference
    - generate_from_pils(pil_images, ...) helper for apps
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        # Tokenizer & T5 model
        model_name = getattr(cfg, 'model', None) or cfg.get('model', 'google/mt5-small')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

        # CLIP encoder (kept on CPU by default; move module.to(device) outside)
        clip_name = getattr(cfg, 'clip_encoder', None) or cfg.get('clip_encoder', 'ViT-B/32')
        self.clip, self.clip_preprocess = clip.load(clip_name, device='cpu')

        # Freezing/unfreezing strategy for CLIP
        freeze_clip = bool(getattr(cfg, 'freeze_clip', None) if hasattr(cfg, 'freeze_clip') else cfg.get('freeze_clip', False))
        unfreeze_last_n = int(getattr(cfg, 'unfreeze_clip_last_n', None) if hasattr(cfg, 'unfreeze_clip_last_n') else cfg.get('unfreeze_clip_last_n', 0))
        if unfreeze_last_n and unfreeze_last_n > 0:
            # Freeze all, then unfreeze last N vision blocks
            for p in self.clip.parameters():
                p.requires_grad = False
            blocks = list(self.clip.visual.transformer.resblocks)
            for block in blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            # Fully freeze or fully trainable
            for p in self.clip.parameters():
                p.requires_grad = not freeze_clip

        # T5 freeze toggles
        if bool(getattr(cfg, 'freeze_t5_encoder', None) if hasattr(cfg, 'freeze_t5_encoder') else cfg.get('freeze_t5_encoder', False)):
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        if bool(getattr(cfg, 'freeze_t5_decoder', None) if hasattr(cfg, 'freeze_t5_decoder') else cfg.get('freeze_t5_decoder', False)):
            for p in self.model.decoder.parameters():
                p.requires_grad = False

        self.prefix_tokens = int(getattr(cfg, 'prefix_tokens', None) if hasattr(cfg, 'prefix_tokens') else cfg.get('prefix_tokens', 8))
        self.use_clip_patch_tokens = bool(getattr(cfg, 'use_clip_patch_tokens', None) if hasattr(cfg, 'use_clip_patch_tokens') else cfg.get('use_clip_patch_tokens', True))

        # Determine CLIP embedding dim for projection
        with torch.no_grad():
            embed_dim = getattr(self.clip.visual, 'output_dim', None)
            if embed_dim is None:
                dummy = torch.randn(1, 3, 224, 224)
                embed_dim = int(self.clip.encode_image(dummy).shape[-1])
        self.proj = ProjectionMLP(in_dim=embed_dim, out_dim=self.model.config.d_model, prefix_tokens=self.prefix_tokens)

        self._cached_sentinel_ids = None

    # ====== Internal helpers ======
    def _encode_image_single(self, images: torch.Tensor) -> torch.Tensor:
        # (B, D) pooled embedding via CLIP encode_image
        return self.clip.encode_image(images)

    def _encode_image_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Patch-token path: average transformer patch tokens (excluding CLS), project to joint space.
        Returns a tensor shaped (B, output_dim) matching encode_image output dim.
        """
        visual = self.clip.visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_tokens = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in visual.transformer.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2)
        patches = x[:, 1:, :]
        pooled = patches.mean(dim=1)
        if hasattr(visual, 'ln_post'):
            pooled = visual.ln_post(pooled)
        if hasattr(visual, 'proj') and visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled

    def _prepare_prefix(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(next(self.parameters()).device)
        clip_emb = self._encode_image_patch_tokens(images) if self.use_clip_patch_tokens else self._encode_image_single(images)
        return self.proj(clip_emb)

    def _get_sentinel_bad_words(self, n: int = 50):
        if self._cached_sentinel_ids is None:
            ids = [self.tokenizer(f'<extra_id_{i}>').input_ids[0] for i in range(n)]
            self._cached_sentinel_ids = [[i] for i in ids]
        return self._cached_sentinel_ids

    # ====== Training forward ======
    def forward(self, images: torch.Tensor, src_texts: Iterable[str], tgt_texts: Iterable[str]):
        device = next(self.parameters()).device
        images = images.to(device)
        clip_emb = self._encode_image_patch_tokens(images) if self.use_clip_patch_tokens else self._encode_image_single(images)
        prefix_emb = self.proj(clip_emb)
        tok_src = self.tokenizer(list(src_texts), return_tensors='pt', padding=True, truncation=True, max_length=getattr(self.cfg, 'src_max_len', None) or self.cfg.get('src_max_len', 64)).to(device)
        tok_tgt = self.tokenizer(list(tgt_texts), return_tensors='pt', padding=True, truncation=True, max_length=getattr(self.cfg, 'tgt_max_len', None) or self.cfg.get('tgt_max_len', 64)).to(device)
        text_emb = self.model.encoder.embed_tokens(tok_src.input_ids)
        full_emb = torch.cat([prefix_emb, text_emb], dim=1)
        full_attn = torch.cat([
            torch.ones(prefix_emb.size(0), self.prefix_tokens, dtype=tok_src.attention_mask.dtype, device=device),
            tok_src.attention_mask
        ], dim=1)
        return self.model(inputs_embeds=full_emb, attention_mask=full_attn, labels=tok_tgt.input_ids)

    # ====== Inference ======
    @torch.inference_mode()
    def generate(
        self,
        image_paths: Optional[List[str]] = None,
        images: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        prompt: str = "Bu görüntüyü açıkla: ",
        ban_sentinels: bool = True,
        **gen_kwargs,
    ) -> List[str]:
        device = next(self.parameters()).device
        num_beams = num_beams or (getattr(self.cfg, 'num_beams_infer', None) or self.cfg.get('num_beams_infer', 4))
        max_new_tokens = max_new_tokens or (getattr(self.cfg, 'max_new_tokens_infer', None) or self.cfg.get('max_new_tokens_infer', 32))
        if images is None:
            assert image_paths is not None, "Provide image_paths or images tensor"
            preprocess = self.clip_preprocess
            from PIL import Image
            pil_images = [Image.open(p).convert('RGB') for p in image_paths]
            images = torch.stack([preprocess(im) for im in pil_images])
        images = images.to(device)
        prefix_tokens = self._prepare_prefix(images)
        tok = self.tokenizer([prompt]*images.size(0), return_tensors='pt', padding=True, truncation=True, max_length=(getattr(self.cfg, 'src_max_len', None) or self.cfg.get('src_max_len', 64))).to(device)
        text_emb = self.model.encoder.embed_tokens(tok.input_ids)
        full_emb = torch.cat([prefix_tokens, text_emb], dim=1)
        full_attn = torch.cat([
            torch.ones(images.size(0), self.prefix_tokens, device=device, dtype=tok.attention_mask.dtype),
            tok.attention_mask
        ], dim=1)
        bad_words_ids = self._get_sentinel_bad_words() if ban_sentinels else None
        gen_ids = self.model.generate(
            inputs_embeds=full_emb,
            attention_mask=full_attn,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            bad_words_ids=bad_words_ids,
            **gen_kwargs,
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_ids]

    @torch.inference_mode()
    def generate_from_pils(
        self,
    pil_images: List[Any],
        prompt: str = "Bu görüntüyü açıkla: ",
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        ban_sentinels: bool = True,
        **gen_kwargs,
    ) -> List[str]:
        images = torch.stack([self.clip_preprocess(im.convert('RGB')) for im in pil_images])
        return self.generate(images=images, prompt=prompt, num_beams=num_beams, max_new_tokens=max_new_tokens, ban_sentinels=ban_sentinels, **gen_kwargs)
