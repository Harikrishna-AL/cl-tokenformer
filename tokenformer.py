import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
import numpy as np

from vit_pytorch import PattentionLayer, TokenformerViT

# Try to import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging. To install: pip install wandb")

# --- Helper Functions ---

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def distributional_loss(current_params, anchor_params):
    anchor_params = anchor_params.detach()
    mean_loss = F.mse_loss(current_params.mean(), anchor_params.mean())
    std_loss = F.mse_loss(current_params.std(), anchor_params.std())
    return mean_loss + std_loss

def apply_grad_mask_hook(grad, mask):
    return grad * mask

# --- Model Architecture ---




# --- Data and Evaluation ---

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in task_classes]
        all_past_test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label < end_class]
        train_subset = Subset(full_train_dataset, train_indices)
        test_subset = Subset(full_test_dataset, all_past_test_indices)
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# --- Training Logic ---

def train_task_final(model, train_loader, optimizer, criterion, device, config, current_task_id):
    model.train()
    epochs, lambda_dist = config["epochs_per_task"], config["lambda_dist"]
    
    hook_handles = []
    if current_task_id > 0:
        for module in model.modules():
            if isinstance(module, PattentionLayer):
                if module.key_param_tokens.requires_grad:
                    h1 = module.key_param_tokens.register_hook(lambda grad, m=module.key_grad_mask: apply_grad_mask_hook(grad, m))
                    hook_handles.append(h1)
                if module.value_param_tokens.requires_grad:
                    h2 = module.value_param_tokens.register_hook(lambda grad, m=module.value_grad_mask: apply_grad_mask_hook(grad, m))
                    hook_handles.append(h2)

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True, desc=f"Task {current_task_id} | Epoch {epoch+1}/{epochs}")
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_cls = criterion(output, target)
            
            loss_dist = 0.0
            if current_task_id > 0:
                for module in model.modules():
                    # Apply regularization to backbone layers only
                    if isinstance(module, PattentionLayer) and module.split_idx is not None:
                        old_keys, new_keys = module.key_param_tokens[:module.split_idx], module.key_param_tokens[module.split_idx:]
                        loss_dist += distributional_loss(new_keys, old_keys)
                        
                        old_values, new_values = module.value_param_tokens[:module.split_idx], module.value_param_tokens[module.split_idx:]
                        loss_dist += distributional_loss(new_values, old_values)
            
            total_loss = loss_cls + lambda_dist * loss_dist
            total_loss.backward()
            optimizer.step()
            loop.set_postfix(loss=total_loss.item(), cls=loss_cls.item(), dist_reg=f"{loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist:.4f}")

    for handle in hook_handles: handle.remove()
    return optimizer

# --- Main Execution ---

if __name__ == '__main__':
    config = {
        "num_tasks": 5, "classes_per_task": 2, "batch_size": 64,
        "epochs_per_task": 10, "lr": 1e-4, "lambda_dist": 10.0,
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if WANDB_AVAILABLE: wandb.init(project="tokenformer-final-cl", config=config)

    model = TokenformerViT(
        image_size=28, patch_size=7, classes_per_task=config["classes_per_task"],
        channels=1, dim=128, depth=2, heads=4, mlp_dim=256, device=DEVICE
    ).to(DEVICE)

    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print(f"Initial model parameters (all trainable): {count_parameters(model):,}")

    all_accuracies = []
    for task_id in range(config["num_tasks"]):
        print(f"\n--- 🚀 Starting Task {task_id} ---")
        
        if task_id > 0:
            model.grow_backbone()
            model.expand_classifier()
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            print(f"Trainable parameters for this task: {count_parameters(model):,}")

        optimizer = train_task_final(model, train_loaders[task_id], optimizer, criterion, DEVICE, config, task_id)
        
        accuracy = evaluate(model, test_loaders[task_id], DEVICE)
        print(f"📊 Accuracy on all seen classes after Task {task_id}: {accuracy:.2f}%")
        all_accuracies.append(accuracy)
        if WANDB_AVAILABLE: wandb.log({"average_accuracy": accuracy, "task_id": task_id})

    print("\n--- Final Evaluation Summary ---")
    for task_id, acc in enumerate(all_accuracies):
        print(f"Average accuracy after finishing Task {task_id}: {acc:.2f}%")