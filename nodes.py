"""
StyleGAN2 Explorer — ComfyUI Custom Nodes
==========================================
Drop the entire stylegan2_explorer/ folder into:
    ComfyUI/custom_nodes/stylegan2_explorer/

Then restart ComfyUI. You'll find three new nodes under the
"StyleGAN2" category in the node menu.

Requirements (installed into ComfyUI's Python environment):
    pip install ninja
    # stylegan2-ada-pytorch must be cloned somewhere and its
    # path set in STYLEGAN2_REPO_PATH below, OR placed at
    # ComfyUI/custom_nodes/stylegan2-ada-pytorch

Usage:
    [StyleGAN2 Model Loader] → [StyleGAN2 Sampler] → PreviewImage / SaveImage
    [StyleGAN2 Model Loader] → [StyleGAN2 Latent Walk] → PreviewImage / SaveImage
"""

import os, sys, glob, pickle, io
import numpy as np
import torch
import folder_paths  # ComfyUI built-in

# ── Locate stylegan2-ada-pytorch repo ────────────────────────────────────────
# Tries a few common locations. Override by setting STYLEGAN2_REPO env var.
_SEARCH_PATHS = [
    os.environ.get("STYLEGAN2_REPO", ""),
    os.path.join(os.path.dirname(__file__), "..", "stylegan2-ada-pytorch"),
    os.path.join(os.path.dirname(__file__), "stylegan2-ada-pytorch"),
    os.path.expanduser("~/stylegan2-ada-pytorch"),
    "C:/stylegan2-ada-pytorch",
    "C:/Users/Public/stylegan2-ada-pytorch",
]

_REPO_PATH = None
for p in _SEARCH_PATHS:
    if p and os.path.isfile(os.path.join(p, "legacy.py")):
        _REPO_PATH = p
        break

if _REPO_PATH is None:
    print(
        "\n[StyleGAN2] ⚠️  Could not find stylegan2-ada-pytorch repo.\n"
        "  Clone it with:\n"
        "    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git\n"
        "  Then either:\n"
        "    • Place it at ComfyUI/custom_nodes/stylegan2-ada-pytorch\n"
        "    • Set the STYLEGAN2_REPO environment variable to its path\n"
    )
else:
    if _REPO_PATH not in sys.path:
        sys.path.insert(0, _REPO_PATH)
    print(f"[StyleGAN2] ✅ Repo found at: {_REPO_PATH}")


# ── Model cache — avoids reloading the same .pkl repeatedly ──────────────────
_model_cache: dict = {}   # path → G_ema tensor


def _load_generator(pkl_path: str, device: torch.device):
    """Load (or return cached) StyleGAN2 generator from a .pkl file."""
    if pkl_path in _model_cache:
        return _model_cache[pkl_path]

    try:
        import legacy  # from stylegan2-ada-pytorch
    except ImportError:
        raise RuntimeError(
            "[StyleGAN2] Cannot import 'legacy'. "
            "Make sure stylegan2-ada-pytorch is on the Python path."
        )

    print(f"[StyleGAN2] Loading model: {pkl_path}")
    with open(pkl_path, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    _model_cache[pkl_path] = G
    print(f"[StyleGAN2] ✅ Loaded z_dim={G.z_dim}  res={G.img_resolution}")
    return G


def _generate(G, z: np.ndarray, truncation_psi: float, device: torch.device) -> torch.Tensor:
    """
    Run the generator for a single z vector.
    Returns a ComfyUI-compatible IMAGE tensor: [1, H, W, 3] float32 0–1
    """
    with torch.no_grad():
        zt = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        lbl = torch.zeros([1, G.c_dim], device=device)
        img = G(zt, lbl, truncation_psi=truncation_psi, noise_mode="const")
        # img is [1, 3, H, W] in range [-1, 1]
        img = (img.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1)
    return img.cpu().float()  # [1, H, W, 3]


def _slerp(z1: np.ndarray, z2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation — smooth paths through latent space."""
    z1n = z1 / (np.linalg.norm(z1) + 1e-8)
    z2n = z2 / (np.linalg.norm(z2) + 1e-8)
    dot = np.clip(np.dot(z1n, z2n), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-6:
        return (1 - t) * z1 + t * z2
    return (np.sin((1 - t) * omega) / np.sin(omega)) * z1 + \
           (np.sin(t * omega) / np.sin(omega)) * z2


# ── Walk state — persists across node executions in a session ────────────────
_walk_state: dict = {}   # keyed by pkl_path


def _get_walk_state(pkl_path: str, z_dim: int) -> dict:
    if pkl_path not in _walk_state:
        _walk_state[pkl_path] = {
            "z":       np.random.randn(z_dim),
            "z_target": np.random.randn(z_dim),
            "t":       0.0,
        }
    return _walk_state[pkl_path]


def _reset_walk(pkl_path: str, z_dim: int):
    _walk_state[pkl_path] = {
        "z":        np.random.randn(z_dim),
        "z_target": np.random.randn(z_dim),
        "t":        0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 1 — StyleGAN2 Model Loader
#  Loads a .pkl and passes the path + device downstream.
#  Keeping it separate lets you wire one model into multiple sampler nodes.
# ─────────────────────────────────────────────────────────────────────────────

class StyleGAN2ModelLoader:
    CATEGORY = "StyleGAN2"
    RETURN_TYPES = ("STYLEGAN2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        # Scan for .pkl files in a few sensible locations
        search_roots = [
            folder_paths.get_input_directory(),
            folder_paths.get_output_directory(),
            os.path.expanduser("~/"),
        ]
        found = []
        for root in search_roots:
            found += glob.glob(os.path.join(root, "**/*.pkl"), recursive=True)

        # Deduplicate and keep just filenames relative to input dir for display
        pkl_choices = []
        input_dir = folder_paths.get_input_directory()
        for p in sorted(set(found)):
            try:
                rel = os.path.relpath(p, input_dir)
            except ValueError:
                rel = p   # different drive on Windows
            pkl_choices.append(rel)

        if not pkl_choices:
            pkl_choices = ["(no .pkl files found — place them in ComfyUI/input/)"]

        return {
            "required": {
                "pkl_file": (pkl_choices, {"default": pkl_choices[0]}),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Or paste full path to .pkl here",
                }),
            },
        }

    def load(self, pkl_file: str, custom_path: str = ""):
        # Prefer custom_path if provided
        if custom_path.strip():
            path = custom_path.strip()
        else:
            input_dir = folder_paths.get_input_directory()
            path = os.path.join(input_dir, pkl_file)
            if not os.path.isfile(path):
                path = pkl_file   # try as absolute

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[StyleGAN2] .pkl not found: {path}\n"
                "Place your model in ComfyUI/input/ or paste the full path."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _load_generator(path, device)   # warm up cache
        return ({"path": path, "device": device},)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 2 — StyleGAN2 Sampler
#  Generates one image from a seed or random z vector.
#  Best for: deterministic generation, seed browsing.
# ─────────────────────────────────────────────────────────────────────────────

class StyleGAN2Sampler:
    CATEGORY = "StyleGAN2"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("STYLEGAN2_MODEL",),
                "seed":            ("INT",   {"default": 0,   "min": 0,   "max": 2**31 - 1}),
                "truncation_psi":  ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "random_seed":     ("BOOLEAN", {"default": False,
                                                "label_on": "Random each run",
                                                "label_off": "Use seed above"}),
            },
        }

    def sample(self, model, seed: int, truncation_psi: float, random_seed: bool):
        G = _load_generator(model["path"], model["device"])

        if random_seed:
            z = np.random.randn(G.z_dim)
        else:
            rng = np.random.RandomState(seed)
            z = rng.randn(G.z_dim)

        image = _generate(G, z, truncation_psi, model["device"])
        return (image,)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 3 — StyleGAN2 Latent Walk
#  Advances a smooth random walk through latent space each time it runs.
#  Best for: exploration, animation, finding interesting regions.
#
#  Connect its output to PreviewImage and hit Queue repeatedly (or use
#  ComfyUI's auto-queue mode) to walk through latent space continuously.
# ─────────────────────────────────────────────────────────────────────────────

class StyleGAN2LatentWalk:
    CATEGORY = "StyleGAN2"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "walk"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("STYLEGAN2_MODEL",),
                "truncation_psi":  ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "step_size":       ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005}),
                "reset_walk":      ("BOOLEAN", {"default": False,
                                                "label_on":  "Reset (new random start)",
                                                "label_off": "Continue walk"}),
            },
        }

    def walk(self, model, truncation_psi: float, step_size: float, reset_walk: bool):
        G = _load_generator(model["path"], model["device"])
        key = model["path"]

        if reset_walk:
            _reset_walk(key, G.z_dim)

        ws = _get_walk_state(key, G.z_dim)

        # Advance
        ws["t"] += step_size
        if ws["t"] >= 1.0:
            ws["z"]        = ws["z_target"].copy()
            ws["z_target"] = np.random.randn(G.z_dim)
            ws["t"]        = 0.0
        else:
            ws["z"] = _slerp(ws["z"], ws["z_target"], ws["t"])

        image = _generate(G, ws["z"], truncation_psi, model["device"])
        return (image,)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 4 — StyleGAN2 Interpolate
#  Smoothly interpolates between two seeds, outputting a batch of frames.
#  Best for: creating transition videos / animation strips.
# ─────────────────────────────────────────────────────────────────────────────

class StyleGAN2Interpolate:
    CATEGORY = "StyleGAN2"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "interpolate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":          ("STYLEGAN2_MODEL",),
                "seed_a":         ("INT",   {"default": 0,   "min": 0, "max": 2**31 - 1}),
                "seed_b":         ("INT",   {"default": 42,  "min": 0, "max": 2**31 - 1}),
                "frames":         ("INT",   {"default": 8,   "min": 2, "max": 64}),
                "truncation_psi": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
            },
        }

    def interpolate(self, model, seed_a: int, seed_b: int, frames: int, truncation_psi: float):
        G = _load_generator(model["path"], model["device"])

        z_a = np.random.RandomState(seed_a).randn(G.z_dim)
        z_b = np.random.RandomState(seed_b).randn(G.z_dim)

        imgs = []
        for i in range(frames):
            t = i / (frames - 1)
            z = _slerp(z_a, z_b, t)
            imgs.append(_generate(G, z, truncation_psi, model["device"]))

        # Stack into [frames, H, W, 3]
        batch = torch.cat(imgs, dim=0)
        return (batch,)


# ─────────────────────────────────────────────────────────────────────────────
#  Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "StyleGAN2ModelLoader":    StyleGAN2ModelLoader,
    "StyleGAN2Sampler":        StyleGAN2Sampler,
    "StyleGAN2LatentWalk":     StyleGAN2LatentWalk,
    "StyleGAN2Interpolate":    StyleGAN2Interpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleGAN2ModelLoader":    "StyleGAN2 Model Loader",
    "StyleGAN2Sampler":        "StyleGAN2 Sampler",
    "StyleGAN2LatentWalk":     "StyleGAN2 Latent Walk",
    "StyleGAN2Interpolate":    "StyleGAN2 Interpolate (batch)",
}
