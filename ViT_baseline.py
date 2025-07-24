import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch import ViT

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging. To install: pip install wandb")


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
        
        train_subset = Subset(full_train_dataset, train_indices)
        test_subset = Subset(full_test_dataset, test_indices)
        
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False))
        print(f"Task {task_id}: Classes {task_classes}, Train samples {len(train_subset)}, Test samples {len(test_subset)}")
        
    return train_loaders, test_loaders

def train_task(model, train_loader, optimizer, criterion, device, task_id, epochs, classes_per_task, global_step):
    """ Trains the model on a single task without any CL strategies. """
    model.train()
    
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{epochs}", leave=False)
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            
            # Map labels to [0, 1] for the current task's head
            target = target - task_id * classes_per_task
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if WANDB_AVAILABLE:
                wandb.log({"loss": loss.item(), "task_id": task_id, "global_step": global_step})
                
            loop.set_postfix(loss=loss.item())
    return global_step

def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    """ Evaluates the model on all tasks seen so far. """
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            # Get the correct test loader for the task being evaluated
            current_test_loader = test_loaders[task_id]
            
            for data, target in current_test_loader:
                data, target = data.to(device), target.to(device)
                
                # Map labels to [0, 1] to match model output
                target = target - task_id * classes_per_task
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
    return accuracies

# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # --- Hyperparameters ---
    config = {
        "num_tasks": 5,
        "classes_per_task": 2,
        "batch_size": 64,
        "epochs_per_task": 3,
        "lr": 1e-4,
        "dim": 128,
        "depth": 2,
        "heads": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
        "emb_dropout": 0.1,
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- W&B Initialization ---
    if WANDB_AVAILABLE:
        wandb.init(
            project="vit-split-mnist-baseline",
            config=config
        )

    # --- Model Configuration ---
    # The model has a single head with `classes_per_task` outputs.
    model = ViT(
        image_size=28,
        patch_size=7,
        num_classes=config["classes_per_task"],
        channels=1, # MNIST is grayscale
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        dropout=config["dropout"],
        emb_dropout=config["emb_dropout"]
    ).to(DEVICE)
    print(f"Model Parameters: {count_parameters(model):,}")

    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()

    # --- Main Continual Learning Loop ---
    all_task_accuracies = []
    global_step = 0
    for task_id in range(config["num_tasks"]):
        print(f"\n--- Training on Task {task_id} ---")
        
        # Re-initialize the optimizer for each task for a standard fine-tuning baseline
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        
        global_step = train_task(
            model, train_loaders[task_id], optimizer, criterion, DEVICE, 
            task_id, config["epochs_per_task"], config["classes_per_task"], global_step
        )
        
        print(f"--- Evaluating after Task {task_id} ---")
        # Evaluate on all tasks seen so far
        num_tasks_seen = task_id + 1
        accuracies = evaluate(model, test_loaders, DEVICE, num_tasks_seen, config["classes_per_task"])
        all_task_accuracies.append(accuracies)
        
        # Log results
        avg_accuracy = np.mean(accuracies)
        print(f"📊 Average Accuracy across {num_tasks_seen} tasks: {avg_accuracy:.2f}%")
        
        if WANDB_AVAILABLE:
            log_dict = {"average_accuracy": avg_accuracy, "global_step": global_step}
            for i, acc in enumerate(accuracies):
                print(f"Accuracy on Task {i}: {acc:.2f}%")
                log_dict[f"acc_task_{i}"] = acc
            wandb.log(log_dict)

    # --- Final Summary ---
    print("\n--- Final Results ---")
    final_avg_acc = np.mean(all_task_accuracies[-1])
    print(f"Final Average Accuracy: {final_avg_acc:.2f}%")
    
    if WANDB_AVAILABLE:
        wandb.summary["final_average_accuracy"] = final_avg_acc
        wandb.finish()
