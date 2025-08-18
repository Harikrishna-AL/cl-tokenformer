# tokenformer_inference.py
import torch
import torch.nn as nn
from vit_pytorch import ContinualLearner, PattentionLayer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse # ### NEW: For command-line arguments ###

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging. To install: pip install wandb")

### --- NEW: CHECKPOINTING FUNCTIONS --- ###

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves model and training state."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """Loads model and training state, handling model growth."""
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        start_task_idx = checkpoint['current_task_id'] + 1
        global_step = checkpoint['global_step']
        results_history = checkpoint['results_history']
        
        # IMPORTANT: Grow the model to the saved size BEFORE loading the state dict
        if checkpoint['current_task_id'] > 0:
            print(f"Growing model to saved state (Task {checkpoint['current_task_id']})...")
            for _ in range(checkpoint['current_task_id']):
                model.grow()

        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Re-initialize optimizer for the new set of trainable parameters
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"=> Loaded checkpoint! Resuming from Task {start_task_idx}")
        return model, optimizer, start_task_idx, global_step, results_history
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return model, optimizer, 0, 0, {}

### --- NEW: RESULTS TABLE FUNCTION --- ###

def print_results_table(history, num_tasks):
    """Prints a formatted table of average accuracies."""
    print("\n\n--- Final Results Summary ---")
    
    # Header
    header = f"{'After Training Task':<25}"
    for i in range(num_tasks):
        header += f"  Task {i} Acc (%) "
    header += "  Average Acc (%)"
    print(header)
    print("-" * len(header))

    # Rows
    for trained_task_id, accs in history.items():
        row = f"{f'Task {trained_task_id}':<25}"
        
        # Print accuracies for tasks seen so far
        for i in range(len(accs)):
            row += f"    {accs[i]:<10.2f}"
        
        # Fill in the rest with dashes
        for i in range(num_tasks - len(accs)):
            row += f"    {'--':<10}"
            
        avg_acc = np.mean(accs)
        row += f"    {avg_acc:<10.2f}"
        print(row)
    print("-" * len(header))


# ... (count_parameters, get_split_mnist_loaders, evaluate, apply_masks_and_hooks, calculate_orthogonality_loss are unchanged) ...
# ... (train_until_plateau is also unchanged from the last corrected version) ...
def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    """ Task-free evaluation for the new single-output model. """
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            start_class = task_id * classes_per_task
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                target = target - start_class
                # Call model in task-free inference mode
                # output = model(data, task_id=None, num_tasks_seen=num_tasks_seen)
                output = model(data, task_id=task_id)
                
                # Get the predicted GLOBAL class index
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                # Compare the predicted GLOBAL index with the true GLOBAL target
                correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            print(f"Accuracy on Task {task_id}: {accuracy:.2f}%")
    return accuracies

def apply_grad_mask_hook(grad, mask):
    return grad * mask

def apply_masks_and_hooks(model, current_task_id, previous_handles):
    for handle in previous_handles:
        handle.remove()
    new_handles = []
    for module in model.growing_transformer.modules():
        if isinstance(module, PattentionLayer):
            new_handles.append(module.key_param_tokens.register_hook(
                lambda grad, m=module: apply_grad_mask_hook(grad, m.key_grad_mask)
            ))
            new_handles.append(module.value_param_tokens.register_hook(
                lambda grad, m=module: apply_grad_mask_hook(grad, m.value_grad_mask)
            ))
    return new_handles

def calculate_orthogonality_loss(growing_module):
    ortho_loss = 0.0
    num_layers = 0
    for module in growing_module.modules():
        if isinstance(module, PattentionLayer) and module.growth_indices:
            last_growth_idx = module.growth_indices[-1]
            k_old = module.key_param_tokens[:last_growth_idx]
            k_new = module.key_param_tokens[last_growth_idx:]

            v_old = module.value_param_tokens[:last_growth_idx]
            v_new = module.value_param_tokens[last_growth_idx:]

            if (k_old.numel() > 0 and k_new.numel() > 0) and (v_old.numel() > 0 and v_new.numel() > 0):
                k_old_norm = F.normalize(k_old, p=2, dim=1)
                k_new_norm = F.normalize(k_new, p=2, dim=1)
                v_old_norm = F.normalize(v_old, p=2, dim=1)
                v_new_norm = F.normalize(v_new, p=2, dim=1)
                cosine_sim_matrix = torch.matmul(k_old_norm, k_new_norm.T)
                cosine_sim_matrix_v = torch.matmul(v_old_norm, v_new_norm.T)
                ortho_loss += (torch.mean(cosine_sim_matrix**2) + torch.mean(cosine_sim_matrix_v**2))
                num_layers += 1
    return ortho_loss / num_layers if num_layers > 0 else 0.0

def train_until_plateau(model, current_task_id, train_loader, optimizer, criterion, device,
                        classes_per_task, global_step, config):
    model.train()
    hook_handles = apply_masks_and_hooks(model, current_task_id, [])
    patience = config["patience"]
    min_delta = config["min_delta_loss"]
    lambda_max = config["lambda_max"]
    lambda_min = config["lambda_min"]
    lambda_decay_epochs = config["lambda_decay_epochs"]
    patience_counter = 0
    best_loss = float('inf')
    epoch = 0
    print(f"🚀 Starting training for model task {current_task_id} (patience={patience}, lambda: {lambda_max} -> {lambda_min}).")
    while patience_counter < patience:
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0.0
        num_batches = 0
        decay_factor = max(0, (1 - epoch / lambda_decay_epochs))
        # current_lambda = lambda_min + (lambda_max - lambda_min) * decay_factor
        current_lambda = lambda_max
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            target = target - current_task_id * classes_per_task
            optimizer.zero_grad()
            output = model(data, current_task_id)
            task_loss = criterion(output, target)
            ortho_loss = 0.0
            if current_task_id > 0:
                ortho_loss = calculate_orthogonality_loss(model.growing_transformer)
            total_loss = task_loss + current_lambda * ortho_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            num_batches += 1
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
        epoch += 1
        if patience_counter >= patience:
            break
    print(f"🏁 Loss plateaued after {epoch} epochs.")
    for handle in hook_handles:
        handle.remove()
    return optimizer, global_step

if __name__ == '__main__':
    # ### NEW: Argument Parser for resuming ###
    parser = argparse.ArgumentParser(description='Tokenformer Continual Learning')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    config = {
        "num_tasks": 5,
        "classes_per_task": 2,
        "batch_size": 32,
        "patience": 2,
        "min_delta_loss": 0.01,
        "lr": 1e-4,
        "lambda_max": 100.0,
        "lambda_min": 0.01,
        "lambda_decay_epochs": 4,
        "attention_bonus": 2.5,
        "data_task_idx": 0
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if WANDB_AVAILABLE:
        wandb.init(project="tokenformer-resnet-cl", config=config)

    model = ContinualLearner(
        dim=128, depth=2, heads=4, mlp_dim=256,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        device=DEVICE,
        attention_bonus=config["attention_bonus"],
    ).to(DEVICE)
    
    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])

    # ### NEW: Initialize state variables and load checkpoint if provided ###
    start_task_idx = 0
    global_step = 0
    current_task_id = 0
    results_history = {}

    if args.resume:
        model, optimizer, start_task_idx, global_step, results_history = load_checkpoint(model, optimizer, args.resume)
        current_task_id = start_task_idx - 1
    
    print(f"Total model parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    # ### MODIFIED: Main loop starts from the correct task index ###
    for data_task_idx in range(start_task_idx, config["num_tasks"]):
        print(f"\n--- Presenting Data from Task {data_task_idx} (Model is on Task {current_task_id}) ---")
        config["data_task_idx"] = data_task_idx

        if data_task_idx > current_task_id and data_task_idx > 0:
             current_task_id += 1
             model.grow()
             print(f"Trainable parameters after growth: {count_parameters(model, trainable_only=True):,}")
             optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
             if WANDB_AVAILABLE:
                 wandb.log({"growth_event": 1, "model_task_id": current_task_id, "global_step": global_step, "trainable_parameters": count_parameters(model, trainable_only=True)})
        
        optimizer, global_step = train_until_plateau(
            model, current_task_id, train_loaders[data_task_idx], optimizer, criterion, DEVICE, 
            config["classes_per_task"], global_step, config
        )
        
        print(f"--- Finished Training on Data Task {data_task_idx} ---")
        accuracies = evaluate(model, test_loaders, DEVICE, current_task_id + 1, config["classes_per_task"])
        
        # ### NEW: Store results and save checkpoint ###
        results_history[current_task_id] = accuracies
        
        state_to_save = {
            'current_task_id': current_task_id,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results_history': results_history,
        }
        save_checkpoint(state_to_save, filename=f"checkpoint_task_{current_task_id}_final.pth.tar")

    # ### NEW: Print final results table ###
    print_results_table(results_history, config["num_tasks"])

    if WANDB_AVAILABLE:
        wandb.summary["final_average_accuracy"] = np.mean(results_history.get(config["num_tasks"] - 1, [0]))
        wandb.summary["final_trainable_parameters"] = count_parameters(model, trainable_only=True)
        wandb.finish()