import torch
import io
import base64
import gc
from PIL import Image
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

# --- 1. SETUP FASTAPI ---
app = FastAPI(title="Fluxer AI Engine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RES = 1024 

# --- 2. RTX 50-SERIES HARDWARE OPTIMIZATIONS ---
print("⚙️ Unlocking Tensor Cores for RTX 50-Series...")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print(f"🧠 Loading FLUX.2-klein-4B into VRAM on {DEVICE}... This will take a moment.")
pipeline_dtype = torch.bfloat16

# --- 3. LOAD THE AI PIPELINES ---
pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=pipeline_dtype,
    use_safetensors=True
)

# Enable VRAM offloading for 16GB cards
pipeline_t2i.to(DEVICE)

# Connect the image-to-image pipeline AFTER offloading
pipeline_i2i = AutoPipelineForImage2Image.from_pipe(pipeline_t2i)

print("⚡ Compiling Transformer for maximum speed...")
print("   (Note: The VERY FIRST image generated will take 1-3 minutes to compile. Everything after will be instant!)")
# FLUX uses a transformer, not a UNet

print("✅ Engine Ready! Listening on http://127.0.0.1:8000")

# --- 4. HELPER FUNCTIONS ---
def prep_image(img, target_w, target_h):
    """Forces image dimensions to be exact multiples of 32 to prevent VAE crashes"""
    img.thumbnail((target_w, target_h))
    w, h = img.size
    new_w = w - (w % 32)
    new_h = h - (h % 32)
    return img.resize((new_w, new_h), Image.LANCZOS)

# --- 5. GODMODE SCHEMA ---
class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "generate"
    image_b64: str | None = None
    
    # Optional Godmode Parameters (Ignored by Node.js GUI if not sent)
    steps: int | None = None          # Override generation length
    guidance: float | None = None     # Override CFG scale
    width: int | None = None          # Custom width
    height: int | None = None         # Custom height
    strength: float | None = None     # How much to alter the image (0.1 to 1.0)
    seed: int | None = None           # Lock the RNG for exact reproducibility 
    format: str = "png"               # Output type ("png" or "jpeg")

# --- 6. THE API ENDPOINT ---
@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        # Fallback to safe defaults if Node.js (or user) doesn't provide them
        gen_steps = req.steps if req.steps else (4 if req.mode == "edit" else 4)
        gen_guidance = req.guidance if req.guidance is not None else 0.0
        gen_w = req.width if req.width else DEFAULT_RES
        gen_h = req.height if req.height else DEFAULT_RES
        gen_strength = req.strength if req.strength else 0.65
        output_format = req.format.upper() if req.format.upper() in ["PNG", "JPEG"] else "PNG"

        # Math safety net: Force dimensions to be multiples of 32
        gen_w = gen_w - (gen_w % 32)
        gen_h = gen_h - (gen_h % 32)

        # Handle reproducible seeds
        generator = torch.Generator(device=DEVICE).manual_seed(req.seed) if req.seed is not None else None

        if req.mode == "edit" and req.image_b64:
            print(f"🎨 Godmode Image-to-Image -> Steps: {gen_steps}, Strength: {gen_strength}, Seed: {req.seed}")
            img_data = base64.b64decode(req.image_b64)
            init_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            init_image = prep_image(init_image, gen_w, gen_h)
            
            out = pipeline_i2i(
                prompt=req.prompt, 
                image=init_image, 
                strength=gen_strength, 
                num_inference_steps=gen_steps, 
                guidance_scale=gen_guidance,
                generator=generator
            ).images[0]
        else:
            print(f"🎨 Godmode Text-to-Image -> Res: {gen_w}x{gen_h}, Steps: {gen_steps}, Seed: {req.seed}")
            out = pipeline_t2i(
                prompt=req.prompt, 
                height=gen_h, 
                width=gen_w, 
                num_inference_steps=gen_steps,
                guidance_scale=gen_guidance,
                generator=generator
            ).images[0]

        # Clean VRAM after generation
        torch.cuda.empty_cache()
        gc.collect()

        # Format and send image
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
