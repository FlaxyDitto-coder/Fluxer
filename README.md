A blazing-fast, 16GB-optimized FLUX Text-to-Image engine with a sleek Web GUI. Features local LoRA hot-swapping, structural edge-tracing, dynamic wildcards, and exact-math Tensor Core 4x upscaling—all running offline on your local hardware.

# 🎨 Fluxer Pro Engine 

Fluxer is a highly optimized, production-grade AI image generation server built around the massive FLUX Transformer model. It is specifically engineered to squeeze a 16GB+ AI pipeline into consumer-grade hardware (like an RTX 5060 Ti) without crashing, spilling over into slow system RAM, or bottlenecking. 

It comes with a built-in dark-mode Web GUI, WebSockets for live progress tracking, and a self-healing auto-installer.

## ✨ Features
* **Hyper-Optimized Memory:** Uses native `bfloat16`, Flash Attention, smart CPU offloading, and VAE tiling to prevent 100% VRAM crashes.
* **The "Holy Grail" Windows Fix:** Bypasses the infamous Windows 260-character path limit that normally breaks Triton and Flash Attention, ensuring ultra-fast generation speeds.
* **Exact-Math Tensor Core Upscaling:** Integrates `AuraSR` for local, offline 4x upscaling. Calculates the perfect "seed" resolution backwards so your final image perfectly hits your target resolution without wasting VRAM.
* **Native Structure Control:** Upload an image and toggle "Edge Tracing". Fluxer uses OpenCV Canny Edge detection to force the AI to paint over the exact silhouette of your original photo.
* **Local LoRA Hot-Swapping:** Select "Anime," "Realism," or "Studio Ghibli" from the dashboard. Fluxer downloads the weights once, injects them for your generation, and unloads them instantly to save memory.
* **Dynamic Wildcards:** Supports `{a|b|c}` syntax in your prompts for massive combinatorial batch generation.

## 💻 System Requirements
* **OS:** Windows 11 / 10 (or Linux)
* **GPU:** NVIDIA RTX Graphics Card with at least **16GB VRAM** (e.g., RTX 4080, 5060 Ti). 
* **RAM:** 32GB System RAM recommended.
* **Python:** 3.10 or higher.
* **Storage:** ~25GB of free space (for the FLUX model, upscaler, and LoRAs).

## 🚀 Installation & Usage

Fluxer is designed to be as plug-and-play as possible. It includes a self-healing auto-installer that will download missing dependencies (like `aura-sr`, `cv2`, and `websockets`) the first time you run it.

1. **Install Core PyTorch & Diffusers:**
   Open your terminal and install the absolute core requirements:
   ```bash
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
   pip install diffusers transformers accelerate fastapi uvicorn pydantic pillow
   ```

2. **Run the Engine:**
   Save the code as `fluxer.py` and run it:
   ```bash
   python fluxer.py
   ```
   *Note: On the very first run, it will automatically download the FLUX.2-klein model (~14GB).*

3. **Open the GUI:**
   Once the console says `✅ Engine Ready!`, open your web browser and go to:
   **`http://127.0.0.1:8000/GUI`**

## 📖 How to Use the GUI

* **Standard Generation:** Type a prompt, set your resolution (must be multiples of 32, e.g., `1024x1024` or `896x1152`), and hit Generate.
* **Upscaling:** Type a massive resolution into the Upscale box (e.g., `2048x2048` or `3840x2160`). Fluxer will calculate the math, generate a seed image, and pass it through the Tensor Core upscaler.
* **Structure Override (ControlNet Alternative):** Upload any photo using the "Image Upload" button and check the **Enable Edge Tracing** box. Type a prompt describing what you want the new image to be, and Fluxer will use the edges of your photo as a strict wireframe.
* **Batches & Wildcards:** Check **Enable Wildcards**, set your batches to `4`, and use brackets in your prompt.

### 🎲 Wildcard Example Prompt
Try pasting this into the GUI with Wildcards enabled and Batches set to 4:
> "A highly detailed cinematic portrait of a {cyberpunk | steampunk | high fantasy} {samurai | cyborg | mage}, wearing {ornate glowing armor | a ragged trench coat}, standing in a {rainy neon-lit alleyway | ruined gothic cathedral}. 8k resolution, masterpiece."

## 📡 API Usage (cURL / Node.js)

Fluxer acts as a headless API if you don't want to use the GUI. You can send JSON payloads to the `/generate` endpoint.

**Basic Request:**
```bash
curl -X POST "[http://127.0.0.1:8000/generate](http://127.0.0.1:8000/generate)" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A glowing neon coffee cup", "resolution": "1024x1024", "steps": 4}' \
  -o "output.png"
```

**Godmode Request (Upscaling + LoRA + Batches):**
```bash
curl -X POST "[http://127.0.0.1:8000/generate](http://127.0.0.1:8000/generate)" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A {red|blue|green} sports car", "dynamic_wildcards": true, "batches": 3, "style": "realism", "upscale": "2048x2048"}' \
  -o "car.png"
``` 
