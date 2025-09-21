import os
import io
import json
from typing import Tuple, Dict, Any

import streamlit as st
from PIL import Image
import torch
from collections import defaultdict

# Reuse inference pipeline from inference.ipynb logic
# We implement a light wrapper here to avoid importing from a notebook.

from model import CLIPmT5Pipeline, ProjectionMLP  # Reuse shared pipeline
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# Workaround: suppress Meteor destructor error ('lock' missing) during interpreter shutdown
try:
    def _meteor_noop_del(self):
        # Avoid noisy AttributeError in some pycocoevalcap versions when cleaning up
        return
    Meteor.__del__ = _meteor_noop_del  # type: ignore[attr-defined]
except Exception:
    pass

# --- Ground truth & metrics helpers ---
_PYCOCO_SCORERS: Dict[str, Any] = {}

def _lazy_load_scorers() -> Dict[str, Any]:
    global _PYCOCO_SCORERS
    if 'bleu' not in _PYCOCO_SCORERS:
        _PYCOCO_SCORERS['bleu'] = Bleu(4)
    if 'meteor' not in _PYCOCO_SCORERS:
        try:
            _PYCOCO_SCORERS['meteor'] = Meteor()
        except Exception as e:
            # Meteor requires Java; if unavailable, skip gracefully
            st.warning(f"METEOR disabled (Java required): {e}")
    if 'rouge' not in _PYCOCO_SCORERS:
        _PYCOCO_SCORERS['rouge'] = Rouge()
    # CIDEr disabled
    return _PYCOCO_SCORERS

@st.cache_data(show_spinner=False)
def load_refs_index(json_path: str) -> Dict[str, list]:
    """Return {filename: [ref1, ref2, ...]} from a Flickr8k-style JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = data['images'] if isinstance(data, dict) and 'images' in data else data
    idx: Dict[str, list] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        fname = row.get('filename') or row.get('image') or row.get('img')
        for s in row.get('sentences', []) or []:
            if isinstance(s, dict):
                cap = s.get('raw')
                if cap and str(cap).strip():
                    idx[fname].append(str(cap).strip())
    return dict(idx)

def compute_caption_metrics(pred: str, refs: list) -> Dict[str, float]:
    """Compute BLEU-1..4, METEOR, ROUGE-L for a single prediction vs list of refs.

    Note: CIDEr is disabled per user request.
    """
    refs_clean = [r.strip() for r in (refs or []) if r and str(r).strip()]
    if not refs_clean:
        return {}
    scorers = _lazy_load_scorers()
    gts = {0: refs_clean}
    res = {0: [pred or "<EMPTY>"]}
    out: Dict[str, float] = {}
    try:
        bleu_scores, _ = scorers['bleu'].compute_score(gts, res)
        out['BLEU-1'] = float(bleu_scores[0])
        out['BLEU-2'] = float(bleu_scores[1])
        out['BLEU-3'] = float(bleu_scores[2])
        out['BLEU-4'] = float(bleu_scores[3])
    except Exception as e:
        st.warning(f"BLEU failed: {e}")
    if 'meteor' in scorers:
        try:
            meteor_score, _ = scorers['meteor'].compute_score(gts, res)
            out['METEOR'] = float(meteor_score)
        except Exception as e:
            st.warning(f"METEOR failed: {e}")
    try:
        rouge_score, _ = scorers['rouge'].compute_score(gts, res)
        out['ROUGE-L'] = float(rouge_score)
    except Exception as e:
        st.warning(f"ROUGE-L failed: {e}")
    return out

class CLIPmT5PipelineInfer(CLIPmT5Pipeline):
    def __init__(self, cfg: Dict[str, Any], tgt_device: torch.device):
        super().__init__(cfg)
        # Move T5 to target device; keep CLIP on CPU (preprocess on CPU, projection/encoder can be moved later if desired)
        self.to(tgt_device)
        self.tgt_device = tgt_device
        # Mirror a few convenience attributes from cfg
        self.num_beams_infer = int(cfg.get('num_beams_infer', 4))
        self.max_new_tokens_infer = int(cfg.get('max_new_tokens_infer', 32))
        self.src_max_len = int(cfg.get('src_max_len', 64))

    @torch.inference_mode()
    def generate_from_pils(self, pil_images, prompt: str = "Bu görüntüyü açıkla: ", num_beams: int = None, max_new_tokens: int = None):
        B = len(pil_images)
        num_beams = int(num_beams or self.num_beams_infer)
        max_new_tokens = int(max_new_tokens or self.max_new_tokens_infer)
        images = torch.stack([self.clip_preprocess(im.convert('RGB')) for im in pil_images])
        images = images.to(self.tgt_device)
        prefix = self._prepare_prefix(images)
        tok = self.tokenizer([prompt] * B, return_tensors='pt', padding=True, truncation=True, max_length=self.src_max_len).to(self.tgt_device)
        text_emb = self.model.encoder.embed_tokens(tok.input_ids)
        full_emb = torch.cat([prefix, text_emb], dim=1)
        full_attn = torch.cat([
            torch.ones(B, self.prefix_tokens, device=self.tgt_device, dtype=tok.attention_mask.dtype),
            tok.attention_mask
        ], dim=1)
        gen_ids = self.model.generate(
            inputs_embeds=full_emb,
            attention_mask=full_attn,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            bad_words_ids=self._get_sentinel_bad_words(),
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_ids]

@st.cache_resource(show_spinner=False)
def load_pipeline(checkpoint_path: str, device_str: str = None) -> Tuple[CLIPmT5PipelineInfer, Dict[str, Any]]:
    device_str = device_str or ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)
    bundle = torch.load(checkpoint_path, map_location='cpu')
    if 'cfg' not in bundle:
        raise RuntimeError("Checkpoint missing cfg")
    cfg = dict(bundle['cfg'])
    pipe = CLIPmT5PipelineInfer(cfg, tgt_device=device)
    state = None
    for k in ('model', 'model_state', 'model_state_dict', 'state_dict'):
        if k in bundle:
            state = bundle[k]
            break
    if state is None:
        raise RuntimeError("No weights found in checkpoint")
    pipe.load_state_dict(state, strict=False)
    pipe.eval()
    return pipe, cfg

st.set_page_config(page_title="Turkish Image Captioning", page_icon="🖼️", layout="centered")
st.title("Turkish Image Captioning")

DEFAULT_CKPT = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt")
ckpt_path = st.text_input("Checkpoint path", value=DEFAULT_CKPT)
default_json = os.environ.get("REFS_JSON", "data/flickr8k/tasviret8k_captions.json")
json_path_input = st.text_input("Ground-truth JSON (Flickr8k format)", value=default_json)

col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
with col2:
    num_beams = st.number_input(
        "Beams",
        min_value=1,
        max_value=8,
        value=4,
        help=(
            "Beam search width during generation. Higher beams explore more candidate sentences "
            "and can improve quality, but increase latency and memory. Typical values: 1 (greedy), 2–5."
        ),
    )
    max_new_tokens = st.number_input(
        "Max tokens",
        min_value=8,
        max_value=128,
        value=32,
        help=(
            "Upper bound on the number of tokens the decoder may generate beyond the prompt. "
            "Higher values allow longer, more detailed captions but cost more time and may drift; "
            "lower values are faster and concise but risk truncation."
        ),
    )

prompt = st.text_input("Prompt", value="Bu görüntüyü açıkla: ")
run = st.button("Caption!")

if run:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}")
    elif not uploaded:
        st.warning("Please upload at least one image.")
    else:
        with st.spinner("Loading model and generating captions..."):
            pipe, cfg = load_pipeline(ckpt_path)
            refs_index = {}
            if json_path_input and os.path.isfile(json_path_input):
                try:
                    refs_index = load_refs_index(json_path_input)
                except Exception as e:
                    st.warning(f"Could not load ground truths: {e}")
            pil_images = []
            for f in uploaded:
                try:
                    pil_images.append(Image.open(io.BytesIO(f.read())).convert('RGB'))
                except Exception as e:
                    st.warning(f"Could not read an image: {e}")
            if pil_images:
                caps = pipe.generate_from_pils(pil_images, prompt=prompt, num_beams=num_beams, max_new_tokens=max_new_tokens)
                for up, im, cap in zip(uploaded, pil_images, caps):
                    # Derive a filename for lookup; for uploads, we can rely on the uploaded file name
                    fname = os.path.basename(getattr(up, 'name', ''))
                    refs = refs_index.get(fname, [])

                    # Show image first
                    st.image(im, width='stretch')

                    # Then ground truths above prediction
                    if refs:
                        with st.expander("Ground truths", expanded=True):
                            for i, r in enumerate(refs[:5], 1):
                                st.markdown(f"**GT {i}:** {r}")
                    else:
                        st.info("No ground-truth captions found for this image name in JSON.")

                    # Prediction
                    st.markdown("**Prediction:** " + (cap or "<EMPTY>"))

                    # Metrics (CIDEr disabled)
                    metrics = compute_caption_metrics(cap, refs)
                    if metrics:
                        st.markdown("**Metrics (BLEU, ROUGE-L, METEOR):**")
                        met_line = "  |  ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                        st.write(met_line)
                    st.divider()
            else:
                st.warning("No valid images to process.")
