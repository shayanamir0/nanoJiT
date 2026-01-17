import torch
import os
from torchvision.utils import make_grid, save_image
from nanojit.model import JustImageTransformer

def sample_images(ckpt="results/nanojit_ep50.pt", output="generated_grid.png"):
    config = {
        "img_size": 256, 
        "patch_size": 16, 
        "dim": 768, 
        "depth": 12, 
        "heads": 12, 
        "bottleneck": 128, 
        "num_classes": 10, 
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    model = JustImageTransformer(**config).to(config['device'])
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=config['device']))
        print(f">> Loaded checkpoint {ckpt}")
    else:
        print(f">>> Checkpoint {ckpt} not found. Initializing random weights.")
        
    model.eval()
    B = config['num_classes']
    y = torch.arange(B, device=config['device']) 
    z = torch.randn(B, 3, 256, 256, device=config['device'])
    
    print(">> Sampling...")
    with torch.no_grad():
        for i in range(50):
            t = torch.full((B,), i/50, device=config['device'])
            x_pred = model(z, t, y)
            v_pred = (x_pred - z) / max(1 - i/50, 1e-5)
            z = z + v_pred * (1.0/50)

    images = (z.clamp(-1, 1) + 1) / 2
    grid = make_grid(images, nrow=5)
    save_image(grid, output)
    print(f">>> Saved {output}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "results/nanojit_ep50.pt"
    sample_images(ckpt)