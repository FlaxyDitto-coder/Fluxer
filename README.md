# Fluxer
Use Flux.2-klein-4b locally on you device. Post through powershell or an app to generate an image. This app runs on http://127.0.0.1:8000 on your device(localhost). Fluxer do have some hardware requirements. You need to read the Readme.md to use it.

Sys req: ~13GB of Vram, ~24-32GB of RAM(sometimes required).
My Hardware: RTX 5060 TI 16GB, 32GB of Sys Ram. on my device it runs on 4 steps in 5-10s

Model is from hf.co(hugging face), provided by black-forest-labs.
To run: pip install torch torchvision --index-url https://download.pytorch.org/whl/ cpu(for cpu(sys ram and slow))/cu{128-130}(for cuda)/mps(for apple silicon)
pip install diffusers fastapi pydantic pillow uvicorn xformers

python3 Fluxer.py.

run pip install... when you are using a different device/virtual enviroment

post a request(powershell):

```powershell 
powershell
$body = @{
  prompt = ""
  mode = "generate"
} | ConvertTo-Json
# Use ConvertTo-Json, it is required. you must fill the prompt parameter and mode must be equal to "generate"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/generate" -Method Post -Body $body -ContentType "application/json" -OutFile "generated image path"
```

or

```powershell
function Get-Flux {
    param([string]$prompt)
    
    # 1. Turn off PowerShell's laggy visual progress bar
    $oldProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    
    $body = @{ prompt = $prompt; mode = "generate" } | ConvertTo-Json
    $fileName = "$PWD\flux_$(Get-Date -Format 'HHmmss').png"
    
    Write-Host "🎨 Generating: '$prompt'..." -ForegroundColor Cyan
    Write-Host "⏳ (Your RTX 5060 Ti is drawing the image. This takes ~5 to 10 seconds...)" -ForegroundColor DarkGray
    
    # Send request (will now save instantly once the GPU is done)
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/generate" -Method Post -Body $body -ContentType "application/json" -OutFile $fileName
    
    Write-Host "✅ Done! Opening $fileName..." -ForegroundColor Green
    Invoke-Item $fileName
    
    # 2. Turn the standard progress bar back on for normal Windows tasks
    $ProgressPreference = $oldProgress
}
```

then

```powershell
Get-Flux prompt
```

post a request(python):

```bash
pip install requests
```

```python
import requests

url = 'http://127.0.0.1:8000/generate'
payload = {'prompt': prompt, 'mode': 'generate')

response = requests.post(url, json=payload)

if response.status_code == 200:
  with open("Generated_image.png", wb) as f:
    f.write(response.content)
  print("Image generated successfully.")
else:
  print(f"Failed to generate image. Status: {response.status_code}")
```

post a request(node.js):

```bash
npm install axios
```

```javascript
const axios = require('axios')
const fs = require('fs')

async function generateAndSaveImage() {
  const url = 'http://127.0.0.1:8000/generate'
  const payload = { prompt: prompt, mode: 'generate' }

  try {
    const response = await axios.post(url, payload, {
      responseType: 'arraybuffer'
    });

    fs.writeFileSync('generated_image.png', response.data);
    console.log('Image saved successfully as generated_image.png');
  } catch (error) {
      console.error('Error generating image:', error.message);
    }
}

generateAndSaveImage();
```


Model hugging face page: https://hf.co/black-forest-labs/Flux.2-klein-4b




##### just a shoutout for black-forest-labs for creating this great AI model!!!!!🥳
