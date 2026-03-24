# ComfyUI StyleGAN2 Explorer

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that bring 
StyleGAN2 latent space exploration directly into your workflow graph.

## Features

- 🎲 **StyleGAN2 Sampler** — Generate images from fixed or random seeds
- 🚶 **Latent Walk** — Continuously slerp through latent space with built-in 
  Start/Stop button and FPS control
- 🎞️ **Interpolate (batch)** — Generate N frames between two seeds for animation
- 📦 Native IMAGE tensors — pipe output into upscalers, ControlNet, img2img, etc.

## Installation

### 1. Clone this repo into your custom_nodes folder

cd ComfyUI/custom_nodes
git clone https://github.com/DEIN-USERNAME/comfyui-stylegan2-explorer.git

### 2. Clone the StyleGAN2-ADA-PyTorch repo next to it

cd ComfyUI/custom_nodes
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

### 3. Install ninja (required by StyleGAN2)

ComfyUI\python_embeds\python.exe -m pip install ninja

### 4. Place your .pkl model files

Drop your StyleGAN2 `.pkl` files into `ComfyUI/input/`

### 5. Restart ComfyUI

The 4 nodes appear under the **StyleGAN2** category.

## Nodes

| Node | Description |
|------|-------------|
| **StyleGAN2 Model Loader** | Loads a `.pkl` — connect to all other nodes |
| **StyleGAN2 Sampler** | Single image from seed. Toggle random for exploration |
| **StyleGAN2 Latent Walk** | Slerp step each queue run. Has ▶ Start Walk button |
| **StyleGAN2 Interpolate** | N frames between seed_a and seed_b |

## Example Workflows

### Latent Walk
[Model Loader] → [Latent Walk] → [Preview Image]

Click ▶ Start Walk on the node to run continuously.

### Interpolation to Video
[Model Loader] → [StyleGAN2 Interpolate] → [Video Combine] → output/

### Walk + Upscale
[Model Loader] → [Latent Walk] → [Upscale Image] → [Preview Image]

## Tips

- **Truncation ψ** low (0.4–0.6) = safe/typical · high (0.8–1.2) = diverse/wild
- **Step size** small (0.02–0.05) = smooth walk · large (0.1–0.2) = fast jumps
- Use two **Primitive** nodes set to `randomize` for random seed_a / seed_b

## Requirements

- ComfyUI (Windows portable or manual install)
- NVIDIA GPU with CUDA
- [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- `ninja` pip package

## License

MIT