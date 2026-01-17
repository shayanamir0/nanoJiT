import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid
import copy
import os
import wandb

from nanojit.model import JustImageTransformer
from nanojit.data import loader

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
    def update(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

@torch.no_grad()
def log_samples(model, config, epoch):
    """Generates samples and logs them to WandB"""
    model.eval()
    B = config['num_classes']
    y = torch.arange(B, device=config['device']) 
    z = torch.randn(B, 3, config['img_size'], config['img_size'], device=config['device'])
    steps = 50
    dt = 1.0 / steps
    
    for i in range(steps):
        t = torch.full((B,), i/steps, device=config['device'])
        x_pred = model(z, t, y)
        v_pred = (x_pred - z) / max(1 - i/steps, 1e-5)
        z = z + v_pred * dt

    # denormalize [-1, 1] -> [0, 1]
    images = (z.clamp(-1, 1) + 1) / 2
    
    grid = make_grid(images, nrow=5)
    wandb.log({"generated_samples": wandb.Image(grid, caption=f"Epoch {epoch}")}, commit=False)

def train():
    # JiT-B/16 config
    config = {
        "img_size": 256, 
        "patch_size": 16, 
        "dim": 768,
        "depth": 12, 
        "heads": 12, 
        "bottleneck": 128, 
        "dropout": 0.1,
        "num_classes": 10,
        "batch_size": 12, 
        "lr": 2e-4, 
        "epochs": 50, 
        "device": "cuda"
    }
    
    wandb.init(project="nanojit", config=config)
    os.makedirs("results", exist_ok=True)

    model = JustImageTransformer(**{k:v for k,v in config.items() if k in 
    ['img_size','patch_size','dim','depth','heads','bottleneck','dropout','num_classes']}).to(config['device'])
    ema = EMA(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.95), weight_decay=0)
    dataloader = loader(config['img_size'], config['batch_size'])

    print(">>> Starting NanoJiT Training...")
    
    global_step = 0
    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(dataloader)
        for x, y in pbar:
            x, y = x.to(config['device']), y.to(config['device'])
            
            # logit-normal time sampling
            t = torch.sigmoid(torch.randn(x.shape[0], device=config['device']) * 0.8 - 0.8)
            eps = torch.randn_like(x)
            
            # rectified flow
            z_t = t.view(-1,1,1,1) * x + (1 - t.view(-1,1,1,1)) * eps
            
            # forward
            x_pred = model(z_t, t, y)
            
            # v-loss
            v_pred = (x_pred - z_t) / (1 - t.view(-1,1,1,1)).clamp(min=0.05)
            loss = F.mse_loss(v_pred, x - eps)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            
            # log metrics
            wandb.log({
                "train_loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
            pbar.set_description(f"Ep {epoch+1} | Loss: {loss.item():.4f}")
            global_step += 1

        # log Samples and checkpoint every 5 epochs
        if (epoch+1) % 5 == 0:
            log_samples(ema.model, config, epoch+1)
            torch.save(ema.model.state_dict(), f"results/nanojit_ep{epoch+1}.pt")
            print(f">>> Saved checkpoint and logged samples for Epoch {epoch+1}")
            
    wandb.finish()

if __name__ == "__main__":
    train()