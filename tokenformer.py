# tokenformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from vit_pytorch import ContinualLearner, PattentionLayer
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size, image_size):
    # (This function is unchanged)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start, end = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start, end))
        train_indices = [i for i, lbl in enumerate(train_dataset.targets) if lbl in task_classes]
        test_indices = [i for i, lbl in enumerate(test_dataset.targets) if lbl in task_classes]
        train_loaders.append(DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders

def apply_grad_mask_hook(grad, mask): return grad * mask
def apply_masks_and_hooks(model):
    # (This function is unchanged)
    handles = []
    for module in model.transformer.modules():
        if isinstance(module, PattentionLayer):
            handles.append(module.key_param_tokens.register_hook(lambda g, m=module: apply_grad_mask_hook(g, m.grad_mask)))
            handles.append(module.value_param_tokens.register_hook(lambda g, m=module: apply_grad_mask_hook(g, m.grad_mask)))
    return handles

def evaluate(model, test_loaders, device, num_tasks_seen):
    # (This function is unchanged)
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
    print(f"Eval after Task {num_tasks_seen-1}: Accuracies = {['%.2f' % acc for acc in accuracies]}, Avg: {np.mean(accuracies):.2f}%")

def train_task(model, train_loader, optimizer, criterion, device, config):
    # (This function is unchanged)
    print(f"\n🚀 Starting Training...")
    run_epochs = config["epochs"]
    hooks = apply_masks_and_hooks(model)
    
    model.train()
    for epoch in range(run_epochs):
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}")
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    for handle in hooks: handle.remove()


if __name__ == '__main__':
    config = {
        "num_tasks": 5, "classes_per_task": 2, "num_total_classes": 10,
        "batch_size": 128, "epochs": 5, "lr": 1e-3,
        "num_initial_tokens": 64, "num_specific_tokens_per_task": 64,
        "image_size": 28, "depth": 2, 
        "latent_dim": 1024, "dg_k": 50, # <-- Autoencoder params
        "pretrained_dg_path": "dg_autoencoder_pretrained.pth"
    }
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = ContinualLearner(
        image_size=config["image_size"],
        depth=config["depth"],
        num_initial_tokens=config["num_initial_tokens"], 
        latent_dim=config["latent_dim"], 
        dg_k=config["dg_k"],
        num_total_classes=config["num_total_classes"],
        channels=1, device=DEVICE,
    ).to(DEVICE)
    
    if os.path.exists(config["pretrained_dg_path"]):
        print(f"Loading pre-trained Autoencoder weights from {config['pretrained_dg_path']}...")
        
        # Load the state dict into the model's feature_extractor attribute
        model.feature_extractor.load_state_dict(
            torch.load(config["pretrained_dg_path"], map_location=DEVICE)
        )
        
        print("Freezing pre-trained feature extractor...")
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        print("✅ Pre-trained feature extractor loaded and frozen.")
    else:
        print(f"⚠️ WARNING: Pre-trained weights not found at '{config['pretrained_dg_path']}'.")
    
    train_loaders, test_loaders = get_split_mnist_loaders(
        config["num_tasks"], config["classes_per_task"], config["batch_size"], config["image_size"])
    criterion = nn.CrossEntropyLoss()
    
    for task_id in range(config["num_tasks"]):
        print(f"\n{'='*20} Task {task_id} {'='*20}")
        if task_id > 0:
            print(f"--- Growing transformer for Task {task_id} ---")
            model.grow_transformer(config["num_specific_tokens_per_task"])
        
        # The optimizer will correctly ignore the frozen feature_extractor
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
        
        train_task(model, train_loaders[task_id], optimizer, criterion, DEVICE, config)
        
        evaluate(model, test_loaders, DEVICE, task_id + 1)