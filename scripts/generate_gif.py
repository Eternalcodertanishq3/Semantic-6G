"""Generate an animated GIF showing Graceful Degradation vs Cliff Effect using a high-res patched image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from train import build_model as build_semantic, load_config, select_device
from models import ClassicalImagePipeline

def load_sample_image():
    path = Path(r"C:\Users\Acer\.gemini\antigravity\brain\25ac5bce-d2f7-4e5f-a0bb-8556d1618fe7\test_parrot_1782817243080.jpg")
    return Image.open(path).convert("RGB")

def patchify(img_tensor, patch_size=32):
    """Convert [1, C, H, W] to [N, C, P, P]"""
    _, c, h, w = img_tensor.shape
    patches = img_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(c, -1, patch_size, patch_size)
    return patches.permute(1, 0, 2, 3)

def unpatchify(patches, h, w, patch_size=32):
    """Convert [N, C, P, P] back to [1, C, H, W]"""
    n, c, ph, pw = patches.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    patches = patches.view(grid_h, grid_w, c, ph, pw)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    return patches.view(c, h, w).unsqueeze(0)

def create_gif():
    config = load_config("config.yaml")
    device = select_device("cpu")
    
    # Load Semantic Model
    semantic = build_semantic(config, device)
    ckpt_path = Path(config["semantic"]["checkpoint"])
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        semantic.load_state_dict(ckpt["model_state"])
    semantic.eval()

    # Load Classical Pipeline
    classical_cfg = dict(config["classical"])
    classical_cfg["symbol_power"] = float(config["channel"]["symbol_power"])
    classical = ClassicalImagePipeline.from_dict(classical_cfg, device=device)

    img_pil = load_sample_image().resize((256, 256), Image.LANCZOS)
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
    
    # Patchify into 64 patches of 32x32
    patches = patchify(img_tensor, 32).to(device)
    
    snrs = [20.0, 15.0, 10.0, 5.0, 2.0, 0.0, -2.0, -5.0]
    frames = []
    to_pil = transforms.ToPILImage()
    
    orig_pil = img_pil
    
    with torch.no_grad():
        for snr in snrs:
            print(f"Processing SNR {snr}dB...")
            # Semantic
            sem_recon_patches = semantic(patches, snr)
            sem_recon_tensor = unpatchify(sem_recon_patches, 256, 256, 32)
            sem_pil = to_pil(sem_recon_tensor[0].clamp(0, 1))
            
            # Classical
            class_recon_patches = classical.transmit(patches, snr)
            if class_recon_patches is None:
                class_pil = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
            else:
                class_recon_tensor = unpatchify(class_recon_patches, 256, 256, 32)
                class_pil = to_pil(class_recon_tensor[0].clamp(0, 1))

            # Combine: [Original | Classical | Semantic]
            panel_size = 256
            combined = Image.new('RGB', (panel_size * 3 + 2, panel_size + 40), color='#1e1e1e')
            combined.paste(orig_pil, (0, 40))
            combined.paste(class_pil, (panel_size + 1, 40))
            combined.paste(sem_pil, (panel_size * 2 + 2, 40))
            
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
                
            draw.text((panel_size//2 - 40, 10), "Original", fill="white", font=font)
            draw.text((panel_size + panel_size//2 - 90, 10), f"Classical (SNR: {snr}dB)", fill="#ff4d4d", font=font)
            draw.text((panel_size*2 + panel_size//2 - 90, 10), f"Semantic (SNR: {snr}dB)", fill="#4dff4d", font=font)
            
            frames.append(combined)
            if snr in [20.0, -5.0]:
                for _ in range(4): # Pause at limits
                    frames.append(combined)

    out_path = Path("outputs/cliff_effect_demo.gif")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:] + frames[::-1],
        duration=500,
        loop=0
    )
    print(f"Saved High-Res GIF to {out_path}")

if __name__ == "__main__":
    create_gif()
