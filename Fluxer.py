import os
# 1. THE HOLY GRAIL FIX: Keeps Flash Attention working on Windows
os.makedirs("C:\\tc", exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = "C:\\tc"

import torch
import io
import gc
from typing import Literal
from PIL import Image
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from diffusers import AutoPipelineForText2Image

# --- 1. SETUP FASTAPI ---
app = FastAPI(title="Fluxer AI Engine")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RES = 1024 

# --- 2. RTX HARDWARE OPTIMIZATIONS ---
print("⚙️ Unlocking Tensor Cores...")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"🧠 Loading FLUX in bfloat16...")
pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)

# 3. THE 16GB VRAM SAVER
pipeline_t2i.enable_model_cpu_offload()

# 4. THE 100% FREEZE FIX
pipeline_t2i.vae.enable_tiling()
pipeline_t2i.vae.enable_slicing()

print("✅ Engine Ready! Listening on http://127.0.0.1:8000")

# --- 5. HELPER FUNCTIONS ---
def prep_image(img, target_w, target_h):
    img.thumbnail((target_w, target_h))
    w, h = img.size
    new_w = w - (w % 32)
    new_h = h - (h % 32)
    return img.resize((new_w, new_h), Image.LANCZOS)

# --- 6. LEAN GODMODE SCHEMA ---
class GenerateRequest(BaseModel):
    prompt: str
    
    # Defaults to "generate" if not provided. 
    # If anything else is provided, the API will reject it.
    mode: Literal["generate"] = "generate" 
    
    steps: int | None = None          
    guidance: float | None = None     
    width: int | None = None          
    height: int | None = None         
    seed: int | None = None           
    format: str = "png"               

# --- 7. THE API ENDPOINT ---
@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        # Fallbacks if the request doesn't include these parameters
        gen_steps = req.steps if req.steps else 4
        gen_guidance = req.guidance if req.guidance is not None else 0.0
        gen_w = req.width if req.width else DEFAULT_RES
        gen_h = req.height if req.height else DEFAULT_RES
        output_format = req.format.upper() if req.format.upper() in ["PNG", "JPEG"] else "PNG"

        # Math safety net: Force dimensions to be multiples of 32
        gen_w = gen_w - (gen_w % 32)
        gen_h = gen_h - (gen_h % 32)

        generator = torch.Generator(device=DEVICE).manual_seed(req.seed) if req.seed is not None else None

        print(f"🎨 Generating -> Res: {gen_w}x{gen_h}, Steps: {gen_steps}")
        
        # Core Generation
        out = pipeline_t2i(
            prompt=req.prompt, 
            height=gen_h, 
            width=gen_w, 
            num_inference_steps=gen_steps,
            guidance_scale=gen_guidance,
            generator=generator
        ).images[0]

        # VRAM Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # Format and return the image
        img_byte_arr = io.BytesIO()
        out.save(img_byte_arr, format=output_format, quality=95)
        img_byte_arr.seek(0)
        
        media_type = f"image/{output_format.lower()}"
        return StreamingResponse(img_byte_arr, media_type=media_type)
        
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return Response(content=f"Failed to generate: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
