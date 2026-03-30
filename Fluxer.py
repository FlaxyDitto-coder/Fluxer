import os
import re
import gc
import io
import time
import random
import asyncio
from typing import Literal, Optional
from PIL import Image
from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

# --- 1. CORE OPTIMIZATIONS & FLASH ATTENTION ---
os.makedirs("C:\\tc", exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = "C:\\tc"

if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI(title="Fluxer AI Engine")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RES = 1024

print("🧠 Loading FLUX (bfloat16)...")
pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)

# Link Img2Img Pipeline BEFORE applying memory hooks
pipeline_i2i = AutoPipelineForImage2Image.from_pipe(pipeline_t2i)

# Apply 16GB VRAM Savers & VAE Freeze Fixes
pipeline_t2i.enable_model_cpu_offload()
pipeline_i2i.enable_model_cpu_offload()
pipeline_t2i.vae.enable_tiling()
pipeline_t2i.vae.enable_slicing()

# Auto-Downloadable Styles (100% Local LoRAs)
STYLES = {
    "anime": ("Shakker-Labs/FLUX.1-dev-LoRA-Anime", "lora.safetensors"),
    "realism": ("XLabs-AI/flux-RealismLora", "lora.safetensors"),
    "ghibli": ("cocktailpeanut/Ghibli-Background", "lora.safetensors")
}

progress_state = {"progress": 0, "status": "Idle"}

# --- 2. SCHEMAS ---
class GenerateRequest(BaseModel):
    prompt: str
    mode: Literal["generate"] = "generate"
    steps: int = 4
    guidance: float = 0.0
    width: int = DEFAULT_RES
    height: int = DEFAULT_RES
    seed: Optional[int] = None
    style: Optional[str] = None           
    image: Optional[str] = None           
    upscale: Optional[str] = None         
    dynamic_wildcards: bool = False
    batches: int = 1

# --- 3. HELPER FUNCTIONS ---
def parse_wildcards(prompt: str) -> str:
    """Parses {a|b|c} syntax into a random choice."""
    def replacer(match):
        options = match.group(1).split('|')
        return random.choice(options).strip()
    return re.sub(r'\{([^}]+)\}', replacer, prompt)

def step_callback(pipe, step, timestep, callback_kwargs):
    """Updates WebSocket progress."""
    global progress_state
    progress_state["progress"] = int((step / pipe.num_timesteps) * 100)
    progress_state["status"] = "Generating..."
    return callback_kwargs

# --- 4. ENDPOINTS ---
@app.get("/")
def read_root():
    return Response(content="Fluxer is running")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(progress_state)
            await asyncio.sleep(0.5)
    except:
        pass

@app.get("/GUI", response_class=HTMLResponse)
def get_gui():
    return """
    <html>
    <head>
        <title>Fluxer GUI</title>
        <style>
            body { background: #121212; color: white; font-family: sans-serif; padding: 20px; }
            .control-panel { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
            input, button, select { padding: 10px; border-radius: 5px; border: none; background: #222; color: white; }
            button { background: #007bff; font-weight: bold; cursor: pointer; }
            button:hover { background: #0056b3; }
            #prompt { width: 100%; font-size: 16px; margin-bottom: 10px; }
            #progress { width: 100%; background: #333; height: 20px; border-radius: 10px; margin-top: 10px; }
            #bar { width: 0%; background: #00e676; height: 100%; border-radius: 10px; transition: width 0.2s; }
            #status-text { margin-top: 5px; font-size: 14px; color: #aaa; }
            img { max-width: 100%; margin-top: 20px; border-radius: 10px; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); }
        </style>
    </head>
    <body>
        <h2>🎨 Fluxer Engine (Pro Mode)</h2>
        <input type="text" id="prompt" placeholder="A cinematic shot of a glowing neon coffee cup...">
        
        <div class="control-panel">
            <input type="number" id="width" placeholder="Width" value="1024" style="width: 80px;">
            <input type="number" id="height" placeholder="Height" value="1024" style="width: 80px;">
            
            <select id="style">
                <option value="">No Style</option>
                <option value="anime">Anime</option>
                <option value="realism">Realism</option>
                <option value="ghibli">Studio Ghibli</option>
            </select>
            
            <input type="text" id="upscale" placeholder="Upscale (e.g. 2048x2048)" style="width: 180px;">
            <button onclick="generate()">Generate Image</button>
        </div>

        <div id="progress"><div id="bar"></div></div>
        <div id="status-text">Idle</div>
        
        <img id="result" />
        
        <script>
            const ws = new WebSocket("ws://" + location.host + "/ws");
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                document.getElementById("bar").style.width = data.progress + "%";
                document.getElementById("status-text").innerText = data.status;
            };
            
            async function generate() {
                document.getElementById("bar").style.width = "0%";
                document.getElementById("status-text").innerText = "Initializing...";
                
                const payload = { 
                    prompt: document.getElementById("prompt").value,
                    width: parseInt(document.getElementById("width").value) || 1024,
                    height: parseInt(document.getElementById("height").value) || 1024
                };
                
                const style = document.getElementById("style").value;
                if (style) payload.style = style;
                
                const upscale = document.getElementById("upscale").value;
                if (upscale) payload.upscale = upscale;
                
                try {
                    const res = await fetch("/generate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    
                    if (res.ok) {
                        const blob = await res.blob();
                        document.getElementById("result").src = URL.createObjectURL(blob);
                        document.getElementById("status-text").innerText = "Done!";
                    } else {
                        const errText = await res.text();
                        document.getElementById("status-text").innerText = "Error: " + errText;
                    }
                } catch (e) {
                    document.getElementById("status-text").innerText = "Network Error";
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/generate")
async def generate(req: GenerateRequest):
    global progress_state
    try:
        os.makedirs("output", exist_ok=True)
        final_images = []
        
        for batch_idx in range(req.batches):
            progress_state = {"progress": 0, "status": f"Initializing Batch {batch_idx+1}"}
            
            current_prompt = parse_wildcards(req.prompt) if req.dynamic_wildcards else req.prompt
            
            gen_w = req.width - (req.width % 32)
            gen_h = req.height - (req.height % 32)
            generator = torch.Generator(device=DEVICE).manual_seed(req.seed) if req.seed else None

            # Hot-Swap Local LoRA
            if req.style and req.style in STYLES:
                repo, weight_name = STYLES[req.style]
                pipeline_t2i.load_lora_weights(repo, weight_name=weight_name)
            
            # Generate Image
            if req.image and os.path.exists(req.image):
                init_image = Image.open(req.image).convert("RGB").resize((gen_w, gen_h))
                out = pipeline_i2i(
                    prompt=current_prompt, image=init_image, 
                    num_inference_steps=req.steps, guidance_scale=req.guidance,
                    generator=generator, callback_on_step_end=step_callback
                ).images[0]
            else:
                out = pipeline_t2i(
                    prompt=current_prompt, height=gen_h, width=gen_w, 
                    num_inference_steps=req.steps, guidance_scale=req.guidance,
                    generator=generator, callback_on_step_end=step_callback
                ).images[0]

            # Clear LoRA
            if req.style and req.style in STYLES:
                pipeline_t2i.unload_lora_weights()

            # --- THE EXACT-MATH UPSCALE ALGORITHM ---
            if req.upscale:
                target_w, target_h = map(int, req.upscale.split("x"))
                current_w, current_h = out.size

                # Only run the complex upscaler if the target is actually larger
                if target_w > current_w or target_h > current_h:
                    
                    # 1. Calculate the exact number of 4x passes needed
                    passes = 0
                    prep_w, prep_h = float(target_w), float(target_h)
                    
                    while prep_w > current_w or prep_h > current_h:
                        prep_w /= 4.0
                        prep_h /= 4.0
                        passes += 1
                        
                    # 2. Downscale the original image to the perfect "seed" resolution
                    prep_w, prep_h = int(prep_w), int(prep_h)
                    progress_state["status"] = f"Downscaling seed to {prep_w}x{prep_h}..."
                    out = out.resize((prep_w, prep_h), Image.LANCZOS)
                    
                    # 3. Load AuraSR and run the required number of passes
                    if passes > 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        from aura_sr import AuraSR
                        upscaler = AuraSR.from_pretrained("fal-ai/AuraSR")
                        
                        if out.mode != "RGB":
                            out = out.convert("RGB")
                            
                        for i in range(passes):
                            progress_state["status"] = f"AuraSR Upscaling (Pass {i+1}/{passes})..."
                            out = upscaler.upscale_4x(out)
                            
                        del upscaler
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    # 4. Final safety crop/resize to correct any 1-pixel rounding errors
                    if out.size != (target_w, target_h):
                        out = out.resize((target_w, target_h), Image.LANCZOS)

            # Save the file locally
            filename = f"output/flux_{int(time.time())}_{batch_idx}.png"
            out.save(filename)
            final_images.append(out)

        progress_state = {"progress": 100, "status": "Done"}
        
        # Stream back the final image
        img_byte_arr = io.BytesIO()
        final_images[-1].save(img_byte_arr, format="PNG")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        progress_state = {"progress": 0, "status": f"Error: {str(e)}"}
        torch.cuda.empty_cache()
        gc.collect()
        return Response(content=f"Failed: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
        
