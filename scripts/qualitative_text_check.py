import argparse
import sys
from pathlib import Path

import torch

# Ensure the root path is accessible to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CharVocabulary
from train_text import build_loaders, build_model, seed_everything, select_device, load_config
from evaluate_text import load_checkpoint_if_available


def run_qualitative_check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    
    # Get validation loader
    _, val_loader = build_loaders(config, fake_data=args.fake_data)
    model = build_model(config, device)
    
    checkpoint_path = Path(config["text"]["checkpoint"])
    load_checkpoint_if_available(model, checkpoint_path, device)
    model.eval()
    
    vocab = CharVocabulary()
    snr_levels = [-5.0, 0.0, 10.0, 20.0]
    
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "qualitative_text_samples.txt"
    
    # Extract one batch and take first 6 samples
    batch = next(iter(val_loader))
    samples = batch[:6].to(device)
    
    with open(out_file, "w", encoding="utf-8") as f, torch.no_grad():
        f.write("=== QUALITATIVE TEXT CHECK ===\n\n")
        
        for i in range(len(samples)):
            original_tokens = samples[i].tolist()
            original_text = vocab.decode(original_tokens).replace("\n", "\\n")
            f.write(f"Sample {i+1}:\n")
            f.write(f"Original: {original_text}\n")
            
            for snr in snr_levels:
                # Add batch dim back for single sample inference
                sample_tensor = samples[i].unsqueeze(0)
                logits = model(sample_tensor, snr)
                pred_tokens = logits.argmax(dim=-1)[0].tolist()
                pred_text = vocab.decode(pred_tokens).replace("\n", "\\n")
                
                f.write(f"SNR {snr:5.1f}dB: {pred_text}\n")
            f.write("-" * 80 + "\n\n")
            
    print(f"Qualitative samples saved to {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative text check.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_qualitative_check(parse_args())
