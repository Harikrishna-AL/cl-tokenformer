# tokenformer_inference.py
import torch
import torch.nn as nn
# Import the new ContinualLearner and the PattentionLayer for our hooks
from vit_pytorch import ContinualLearner, PattentionLayer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging. To install: pip install wandb")

def count_parameters(model, trainable_only=False):
    """Counts the total or trainable-only parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    """ Prepares the Split MNIST dataloaders with transformation for ResNet. """
    # --- MODIFIED: Transformation pipeline for ImageNet models ---
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # The rest of the function is the same
    full_train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label in task_classes]
        train_subset, test_subset = Subset(full_train_dataset, train_indices), Subset(full_test_dataset, test_indices)
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False))
        print(f"Task {task_id}: Classes {task_classes}, Train samples {len(train_subset)}, Test samples {len(test_subset)}")
    return train_loaders, test_loaders

# MODIFIED: The model passed in is the top-level ContinualLearner
def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            start_class = task_id * classes_per_task
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                target = target - start_class
                output = model(data, task_id)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            print(f"Accuracy on Task {task_id}: {accuracy:.2f}%")
    return accuracies

def apply_grad_mask_hook(grad, mask):
    return grad * mask

# MODIFIED: Hooks are now applied to the growing module specifically
def apply_masks_and_hooks(model, current_task_id, previous_handles):
    for handle in previous_handles:
        handle.remove()
    
    new_handles = []
    
    # We only apply hooks to the part of the model that grows and has masks
    for module in model.growing_module.modules():
        if isinstance(module, PattentionLayer):
            new_handles.append(module.key_param_tokens.register_hook(
                lambda grad, m=module: apply_grad_mask_hook(grad, m.key_grad_mask)
            ))
            new_handles.append(module.value_param_tokens.register_hook(
                lambda grad, m=module: apply_grad_mask_hook(grad, m.value_grad_mask)
            ))
    return new_handles

# MODIFIED: Pass the growing module to the loss function
def calculate_orthogonality_loss(growing_module):
    ortho_loss = 0.0
    num_layers = 0
    for module in growing_module.modules():
        if isinstance(module, PattentionLayer) and module.growth_indices:
            last_growth_idx = module.growth_indices[-1]
            k_old = module.key_param_tokens[:last_growth_idx]
            k_new = module.key_param_tokens[last_growth_idx:]

            if k_old.numel() > 0 and k_new.numel() > 0:
                k_old_norm = F.normalize(k_old, p=2, dim=1)
                k_new_norm = F.normalize(k_new, p=2, dim=1)
                cosine_sim_matrix = torch.matmul(k_old_norm, k_new_norm.T)
                ortho_loss += torch.mean(cosine_sim_matrix**2)
                num_layers += 1
    return ortho_loss / num_layers if num_layers > 0 else 0.0

### MODIFIED FUNCTION ###
### CORRECTED FUNCTION ###
def train_until_plateau(model, current_task_id, train_loader, optimizer, criterion, device,
                        classes_per_task, global_step, ema_perplexity, config):
    model.train()
    # Note: backbone is in eval mode inside the forward pass
    
    hook_handles = apply_masks_and_hooks(model, current_task_id, [])

    patience = config["patience"]
    min_delta = config["min_delta_loss"]
    # --- Lambda scheduling parameters ---
    lambda_min = config["lambda_min"]
    lambda_max = config["lambda_max"]
    lambda_decay_epochs = config["lambda_decay_epochs"]
    
    patience_counter = 0
    best_loss = float('inf')
    epoch = 0

    print(f"🚀 Starting training for model task {current_task_id} (patience={patience}, lambda_max={lambda_max}).")

    while patience_counter < patience:
        loop = tqdm(train_loader, leave=True)
        
        epoch_loss = 0.0
        num_batches = 0
        
        # --- Calculate current lambda for this epoch ---
        decay_factor = max(0, (1 - epoch / lambda_decay_epochs))
        current_lambda = lambda_min + (lambda_max - lambda_min) * decay_factor  

        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            target = target - current_task_id * classes_per_task
            
            optimizer.zero_grad()
            output = model(data, current_task_id)
            
            task_loss = criterion(output, target)
            ortho_loss = 0.0
            
            if current_task_id > 0:
                ortho_loss = calculate_orthogonality_loss(model.growing_module)

            total_loss = task_loss + current_lambda * ortho_loss
            
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1
            
            # ... logging logic ...
            global_step += 1
            if WANDB_AVAILABLE:
                log_data = {
                    "task_loss": task_loss.item(), "total_loss": total_loss.item(),
                    "model_task_id": current_task_id, "epoch": epoch, "global_step": global_step,
                    "current_lambda": current_lambda
                }
                if current_task_id > 0:
                    log_data["ortho_loss"] = ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss
                wandb.log(log_data)
            
            loop.set_description(f"Data Task {config['data_task_idx']} | Model Task {current_task_id} | Epoch {epoch+1}")
            loop.set_postfix(loss=total_loss.item(), ortho=f"{ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else 0:.4f}", lambda_o=f"{current_lambda:.4f}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        print(f"\nEpoch {epoch+1} ended. Avg Total Loss: {avg_epoch_loss:.4f}. Best Loss: {best_loss:.4f}")

        if avg_epoch_loss < best_loss - min_delta:
            best_loss = avg_epoch_loss
            patience_counter = 0
            print(f"✅ Loss improved. Resetting patience counter.")
        else:
            patience_counter += 1
            print(f"⚠️ Loss did not improve. Patience: {patience_counter}/{patience}")
        
        if WANDB_AVAILABLE:
             wandb.log({
                "avg_epoch_loss": avg_epoch_loss, "best_epoch_loss": best_loss,
                "patience_counter": patience_counter, "global_step": global_step
            })

        epoch += 1
        if patience_counter >= patience:
            break
            
    print(f"🏁 Loss plateaued after {epoch} epochs.")
    for handle in hook_handles:
        handle.remove()
        
    # --- FIX: Return both optimizer and global_step ---
    return optimizer, global_step

if __name__ == '__main__':
    config = {
        "num_tasks": 5,
        "classes_per_task": 2,
        "batch_size": 32, # Smaller batch size for larger ResNet model
        "patience": 2,
        "min_delta_loss": 0.01,
        "lr": 1e-4,
        "lambda_max": 10.0, # Start with a strong orthogonality constraint
        "lambda_min" : 2.5,
        "lambda_decay_epochs": 10, # Decay lambda over 4 epochs
        "data_task_idx": 0
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if WANDB_AVAILABLE:
        wandb.init(project="tokenformer-resnet-cl", config=config)

    # --- MODIFIED: Model Instantiation ---
    model = ContinualLearner(
        dim=128, depth=2, heads=4, mlp_dim=256,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        device=DEVICE
    ).to(DEVICE)

    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    
    # --- MODIFIED: Optimizer only sees trainable parameters ---
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])

    global_step = 0
    current_task_id = 0
    
    print(f"Total model parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    for data_task_idx, train_loader in enumerate(train_loaders):
        print(f"\n--- Presenting Data from Task {data_task_idx} (Model is on Task {current_task_id}) ---")
        config["data_task_idx"] = data_task_idx

        if data_task_idx > 0:
            current_task_id += 1
            model.grow()
            print(f"Trainable parameters after growth: {count_parameters(model, trainable_only=True):,}")
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
            if WANDB_AVAILABLE:
                wandb.log({"growth_event": 1, "model_task_id": current_task_id, "global_step": global_step, "trainable_parameters": count_parameters(model, trainable_only=True)})
        
        optimizer, global_step = train_until_plateau(
            model, current_task_id, train_loader, optimizer, criterion, DEVICE, 
            config["classes_per_task"], global_step, None, config
        )
        
        print(f"--- Finished Training on Data Task {data_task_idx} ---")
        accuracies = evaluate(model, test_loaders, DEVICE, current_task_id + 1, config["classes_per_task"])
        avg_accuracy = np.mean(accuracies)
        print(f"📊 Average Accuracy across {len(accuracies)} seen tasks: {avg_accuracy:.2f}%")
        if WANDB_AVAILABLE:
            wandb.log({"average_accuracy": avg_accuracy, "global_step": global_step})

    print("\n--- Final Evaluation ---")
    final_accuracies = evaluate(model, test_loaders, DEVICE, config["num_tasks"], config["classes_per_task"])
    if WANDB_AVAILABLE:
        wandb.summary["final_average_accuracy"] = np.mean(final_accuracies)
        wandb.summary["final_trainable_parameters"] = count_parameters(model, trainable_only=True)
        wandb.finish()