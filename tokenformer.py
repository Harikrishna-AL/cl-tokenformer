# tokenformer_inference.py
import torch
import torch.nn as nn
from vit_pytorch import TokenformerViT, PattentionLayer # Make sure to import from your local file
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging. To install: pip install wandb")

# ... (get_split_mnist_loaders, evaluate, apply_grad_mask_hook, apply_masks_and_hooks functions remain the same) ...

def count_parameters(model):
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    """ Prepares the Split MNIST dataloaders. """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
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

def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    """ Evaluates the model on all seen tasks and returns a list of accuracies. """
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
    """A gradient hook that applies a mask to the gradients."""
    return grad * mask

def apply_masks_and_hooks(model, current_task_id, previous_handles):
    """Removes old hooks and applies new ones for the current task."""
    for handle in previous_handles:
        handle.remove()
    
    new_handles = []
    
    for module in model.modules():
        if isinstance(module, PattentionLayer):
            new_handles.append(module.key_param_tokens.register_hook(
                lambda grad, m=module: apply_grad_mask_hook(grad, m.key_grad_mask)
            ))
            new_handles.append(module.value_param_tokens.register_hook(
                lambda grad, m=module: apply_grad_mask_hook(grad, m.value_grad_mask)
            ))

    if current_task_id > 0:
        new_handles.append(model.pos_embedding.register_hook(lambda grad: grad * 0))
        new_handles.append(model.cls_token.register_hook(lambda grad: grad * 0))
        for name, param in model.named_parameters():
            if 'transformer' in name and 'norm' in name:
                new_handles.append(param.register_hook(lambda grad: grad * 0))
            if 'to_patch_embedding' in name and not 'weight' in name and not 'bias' in name:
                new_handles.append(param.register_hook(lambda grad: grad * 0))
    return new_handles


### NEW FUNCTION ###
def calculate_orthogonality_loss(model):
    """
    Calculates the orthogonality loss for all Pattention layers in the model.
    This loss encourages the newly added key parameters to be orthogonal to all previous key parameters.
    """
    ortho_loss = 0.0
    for module in model.modules():
        if isinstance(module, PattentionLayer) and module.growth_indices:
            # This layer has grown at least once
            
            # The 'old' parameters are all parameters up to the last growth boundary
            last_growth_idx = module.growth_indices[-1]
            k_old = module.key_param_tokens[:last_growth_idx]
            
            # The 'new' parameters are those added during the last growth phase
            k_new = module.key_param_tokens[last_growth_idx:]

            # Calculate the dot product between old and new key subspaces
            # We want this to be a zero matrix, so we penalize its norm
            if k_old.numel() > 0 and k_new.numel() > 0:
                dot_product_matrix = torch.matmul(k_old, k_new.T)
                target_matrix = torch.zeros_like(dot_product_matrix)
                # This is an AVERAGE of squares
                ortho_loss += F.mse_loss(dot_product_matrix, target_matrix)

    return ortho_loss


### MODIFIED FUNCTION ###
def train_until_plateau(model, current_task_id, train_loader, optimizer, criterion, device,
                        classes_per_task, global_step, ema_perplexity, config):
    """
    Trains the model on a single task's data until the loss plateaus,
    now including the orthogonality loss.
    """
    model.train()
    
    hook_handles = apply_masks_and_hooks(model, current_task_id, [])

    patience = config["patience"]
    min_delta = config["min_delta_loss"]
    lambda_ortho = config["lambda_ortho"] # Get lambda from config
    
    patience_counter = 0
    best_loss = float('inf')
    epoch = 0

    print(f"🚀 Starting training for model task {current_task_id} until plateau (patience={patience}, lambda_ortho={lambda_ortho}).")

    while patience_counter < patience:
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0.
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(loop):
            # --- Regular Training Step ---
            data, target = data.to(device), target.to(device)
            target = target - current_task_id * classes_per_task
            
            optimizer.zero_grad()
            output = model(data, current_task_id)
            
            # --- Calculate Losses ---
            task_loss = criterion(output, target)
            ortho_loss = 0.0
            
            # Add orthogonality loss only for new tasks (after first growth)
            if current_task_id > 0:
                ortho_loss = calculate_orthogonality_loss(model)

            total_loss = task_loss + lambda_ortho * ortho_loss
            
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1
            
            # --- Update EMA Perplexity (based on task_loss) ---
            current_perplexity = torch.exp(task_loss).item()
            if ema_perplexity is None:
                ema_perplexity = current_perplexity
            else:
                ema_perplexity = config["ema_alpha"] * current_perplexity + (1 - config["ema_alpha"]) * ema_perplexity

            # --- Logging ---
            global_step += 1
            if WANDB_AVAILABLE:
                log_data = {
                    "task_loss": task_loss.item(), "total_loss": total_loss.item(),
                    "perplexity": current_perplexity, "ema_perplexity": ema_perplexity,
                    "model_task_id": current_task_id, "epoch": epoch, "global_step": global_step
                }
                if current_task_id > 0:
                    log_data["ortho_loss"] = ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss
                wandb.log(log_data)
            
            loop.set_description(f"Data Task {config['data_task_idx']} | Model Task {current_task_id} | Epoch {epoch+1}")
            loop.set_postfix(loss=total_loss.item(), ortho=f"{ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else 0:.4f}", patience=f"{patience_counter}/{patience}")

        # --- Plateau Check ---
        avg_epoch_loss = epoch_loss / num_batches
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
    
    print(f"🏁 Loss plateaued after {epoch} epochs. Finished training for model task {current_task_id}.")
    for handle in hook_handles:
        handle.remove()
        
    return optimizer, global_step, ema_perplexity


if __name__ == '__main__':
    # --- Hyperparameters ---
    config = {
        "num_tasks": 5,
        "classes_per_task": 2,
        "batch_size": 64,
        "patience": 2,
        "min_delta_loss": 0.01,
        "lr": 1e-4,
        "lambda_ortho": 1e-4,  # ### NEW ###: Strength of the orthogonality constraint
        "perplexity_growth_threshold": 3.0,
        "ema_alpha": 0.1,
        "data_task_idx": 0
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- W&B Initialization ---
    if WANDB_AVAILABLE:
        wandb.init(
            project="tokenformer-split-mnist-ortho", # New project name
            config={
                "learning_rate": config["lr"],
                "patience": config["patience"],
                "min_delta_loss": config["min_delta_loss"],
                "batch_size": config["batch_size"],
                "lambda_ortho": config["lambda_ortho"], # ### NEW ###
                "architecture": "TokenformerViT-Ortho-Grow",
                "dataset": "SplitMNIST",
                "perplexity_growth_threshold": config["perplexity_growth_threshold"],
                "ema_alpha": config["ema_alpha"]
            }
        )

    # --- Model, Data, Optimizer, Criterion ---
    model = TokenformerViT(
        image_size=28, patch_size=7, num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        channels=1, dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1, emb_dropout=0.1, device=DEVICE
    ).to(DEVICE)

    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # --- Main Continual Learning Loop ---
    # ... (The main loop remains exactly the same as your previous version) ...
    global_step = 0
    current_task_id = 0 # This is the model's internal task id, which will be incremented on growth
    ema_perplexity = None # Initialize EMA tracker

    print(f"Initial model parameters: {count_parameters(model):,}")

    for data_task_idx, train_loader in enumerate(train_loaders):
        print(f"\n--- Presenting Data from Task {data_task_idx} (Model is on Task {current_task_id}) ---")
        config["data_task_idx"] = data_task_idx

        ### MODIFIED: Growth check is performed before starting training for a new task ###
        if data_task_idx > 0:
            first_batch_data, first_batch_target = next(iter(train_loader))
            
            # The most reliable signal for growth in Split-MNIST is detecting new class labels
            if torch.max(first_batch_target) >= (current_task_id + 1) * config["classes_per_task"]:
                print(f"🔎 New class labels detected. Growth condition met!")
                current_task_id += 1
                
                model.grow()
                print(f"Parameters after growth: {count_parameters(model):,}")
                
                # Optimizer must be re-initialized to include new parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
                # Reset EMA perplexity as the model and data distribution have changed
                ema_perplexity = None
                
                if WANDB_AVAILABLE:
                    wandb.log({"growth_event": 1, "model_task_id": current_task_id, "global_step": global_step})
                    wandb.log({"model_parameters": count_parameters(model), "global_step": global_step})

        # --- Train on the current task's data until loss plateaus ---
        optimizer, global_step, ema_perplexity = train_until_plateau(
            model, current_task_id, train_loader, optimizer, criterion, DEVICE, 
            config["classes_per_task"], global_step, ema_perplexity, config
        )
        
        print(f"--- Finished Training on Data Task {data_task_idx} ---")
        # Evaluate on all tasks seen so far by the model
        accuracies = evaluate(model, test_loaders, DEVICE, current_task_id + 1, config["classes_per_task"])
        
        avg_accuracy = np.mean(accuracies)
        print(f"📊 Average Accuracy across {len(accuracies)} seen tasks: {avg_accuracy:.2f}%")
        if WANDB_AVAILABLE:
            wandb.log({
                "average_accuracy": avg_accuracy,
                "global_step": global_step
            })

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on All Tasks ---")
    final_accuracies = evaluate(model, test_loaders, DEVICE, config["num_tasks"], config["classes_per_task"])
    print(f"\nFinal model parameters: {count_parameters(model):,}")
    print(f"Average Accuracy: {np.mean(final_accuracies):.2f}%")

    if WANDB_AVAILABLE:
        wandb.summary["final_average_accuracy"] = np.mean(final_accuracies)
        wandb.summary["final_model_parameters"] = count_parameters(model)
        wandb.finish()