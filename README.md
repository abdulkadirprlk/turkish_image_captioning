# Turkish Image Captioning (CLIP ViT-B/32 + mT5-small)

Generate Turkish captions for images using a lightweight multimodal pipeline:
- Image encoder: OpenAI CLIP (ViT‑B/32)
- Prefix projection: learnable MLP that maps image features to T5 encoder tokens
- Text decoder: mT5‑small

This repo includes a Streamlit app for drag‑and‑drop inference and a simple Python API.


## What you’ll need

- Python 3.10+ (Linux, macOS, or Windows)
- Git (required because some dependencies are installed from GitHub)
- A model checkpoint file at `checkpoints/best.pt`
  - You can use your own, download one you trained, or place any compatible checkpoint at that path.
  - Alternatively, set `CHECKPOINT_PATH` to point to your checkpoint.
- Optional but recommended: a GPU (CUDA) or Apple Silicon (MPS). CPU works but will be slower.

Notes on metrics:
- The app can show BLEU, ROUGE‑L, and METEOR. METEOR requires Java (OpenJDK). If Java is missing, METEOR is skipped automatically.


## Option A — Quick start with pip (Streamlit UI)

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows: .venv\Scripts\activate
```

2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Ensure your checkpoint is available

Place your checkpoint at `checkpoints/best.pt` or set an environment variable:

```bash
export CHECKPOINT_PATH=checkpoints/best.pt  # adjust if your file is elsewhere
```

4) (Optional) Ground‑truth JSON for metrics

If you want to see BLEU/ROUGE/METEOR against Flickr8k‑style references, provide a JSON path via the UI or set:

```bash
export REFS_JSON=data/flickr8k/tasviret8k_captions.json
```

5) Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then open the displayed local URL (typically http://localhost:8501), upload images, tweak “Beams” and “Max tokens”, and click “Caption!”.


## Option B — Programmatic inference (Python API)

You can run inference without the UI using the shared pipeline in `model.py`.

```python
import torch
from model import CLIPmT5Pipeline

# Load checkpoint (produced by training in this repo)
ckpt_path = "checkpoints/best.pt"
bundle = torch.load(ckpt_path, map_location="cpu")
cfg = dict(bundle["cfg"])  # training-time config is embedded in the checkpoint

# Build pipeline
pipe = CLIPmT5Pipeline(cfg)

# Load weights (supports multiple key names)
state = None
for k in ("model", "model_state", "model_state_dict", "state_dict"):
	if k in bundle:
		state = bundle[k]
		break
if state is None:
	raise RuntimeError("No weights found in checkpoint")
pipe.load_state_dict(state, strict=False)
pipe.eval()

# Generate a caption from local image paths
images = ["/path/to/image1.jpg", "/path/to/image2.png"]
captions = pipe.generate(
	image_paths=images,
	prompt="Bu görüntüyü açıkla: ",
	num_beams=4,
	max_new_tokens=32,
	ban_sentinels=True,
)
for p, c in zip(images, captions):
	print(p, "->", c)
```

Apple Silicon: If you want to leverage MPS, you can move the model to `mps`:

```python
import torch
if torch.backends.mps.is_available():
	pipe.to(torch.device("mps"))
```


## Option C — Docker (CPU) with docker compose

The provided Docker setup runs on CPU by default and includes both a Jupyter and a Streamlit service.

Prerequisites:
- Docker and Docker Compose installed
- Your checkpoint mounted at `./checkpoints/best.pt`

Start the Streamlit app:

```bash
docker compose up streamlit
```

Open http://localhost:8501 and use the UI. The compose file mounts `./data` and `./checkpoints` read‑only into the container.

Start the Jupyter environment instead:

```bash
docker compose up notebook
```


## Beams and Max tokens (guidance)

- Beams: Beam search width during generation. Higher values explore more candidates (better quality) but are slower and use more memory. Try 1 (greedy), 2–5.
- Max tokens: Upper bound on the number of tokens to generate beyond the prompt. Higher values allow longer captions but can drift; lower values are faster and concise.


## Troubleshooting

- Missing Java / METEOR errors: Install OpenJDK (e.g., `brew install openjdk` on macOS, `sudo apt-get install openjdk-17-jre` on Debian/Ubuntu). The app will still work without METEOR.
- pycocoevalcap build issues: Ensure `git` and compiler tools are installed. On macOS, install Xcode Command Line Tools (`xcode-select --install`). On Windows, install “Build Tools for Visual Studio”.
- Slow CPU inference: Prefer a CUDA GPU or Apple Silicon (MPS). The Streamlit app auto‑selects MPS when available.
- Large downloads: First run will download models from Hugging Face. You can set caches via env vars:
  - `HF_HOME` and `TRANSFORMERS_CACHE` to control cache location.
- Checkpoint not found: Make sure `checkpoints/best.pt` exists or set `CHECKPOINT_PATH`.


## Project layout (key files)

- `model.py` — Shared CLIP+mT5 pipeline used for training and inference
- `streamlit_app.py` — Streamlit UI for local captioning and optional metrics
- `requirements.txt` — Dependencies (includes CLIP and COCO eval from GitHub)
- `checkpoints/best.pt` — Expected checkpoint path for inference
- `data/flickr8k/tasviret8k_captions.json` — Optional references for metrics in the UI


## License and acknowledgements

- CLIP by OpenAI, Transformers by Hugging Face, mT5 by Google.
- COCO caption metrics via `pycocoevalcap`.
