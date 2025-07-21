import torch
import torch.nn as nn
from vit_pytorch import ViT, TokenformerViT, PattentionLayer
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



def count_parameters(model):
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def calculate_frozen_percentage(model):
    """Calculates the percentage of frozen parameters in the model."""
    total_params = count_parameters(model)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if total_params == 0:
        return 0
    return (frozen_params / total_params) * 100

def apply_grad_mask_hook(grad, mask):
    """A gradient hook that applies a mask to the gradients."""
    # print(grad.shape, mask.shape)
    return grad * mask

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

def train_task(model, task_id, train_loader, optimizer, criterion, device, epochs, classes_per_task, global_step):
    """ Trains the model on a single task. """
    model.train()
    start_class = task_id * classes_per_task
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            target = target - start_class
            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            
            # --- Logging to W&B ---
            if WANDB_AVAILABLE:
                perplexity = torch.exp(loss)
                wandb.log({
                    "loss": loss.item(),
                    "perplexity": perplexity.item(),
                    "task_id": task_id,
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            loss.backward()
            optimizer.step()
            global_step += 1
            loop.set_description(f"Task {task_id} Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())
    return global_step

def evaluate(model, test_loaders, device, num_tasks, classes_per_task):
    """ Evaluates the model on all seen tasks. """
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks):
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

if __name__ == '__main__':
    # --- Hyperparameters ---
    NUM_TASKS = 5
    CLASSES_PER_TASK = 2
    BATCH_SIZE = 64
    EPOCHS_PER_TASK = 3
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- W&B Initialization ---
    if WANDB_AVAILABLE:
        wandb.init(
            project="tokenformer-split-mnist",
            config={
                "learning_rate": LR,
                "epochs_per_task": EPOCHS_PER_TASK,
                "batch_size": BATCH_SIZE,
                "architecture": "TokenformerViT-Grow-Mask",
                "dataset": "SplitMNIST",
            }
        )

    # --- Model Configuration ---
    model = TokenformerViT(
        image_size=28, patch_size=7, num_tasks=NUM_TASKS, classes_per_task=CLASSES_PER_TASK,
        channels=1, dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1, emb_dropout=0.1, device=DEVICE
    ).to(DEVICE)

    train_loaders, test_loaders = get_split_mnist_loaders(NUM_TASKS, CLASSES_PER_TASK, BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    # --- Main Continual Learning Loop ---
    hook_handles = []
    global_step = 0
    for task_id in range(NUM_TASKS):
        print(f"\n--- Starting Training on Task {task_id} ---")
        print(f"Parameters before growth: {count_parameters(model):,}")

        # Remove any hooks from the previous iteration for a clean state
        for handle in hook_handles:
            handle.remove()
        hook_handles = []

        if task_id > 0:
            model.grow()
            print(f"Parameters after growth: {count_parameters(model):,}")

        # --- Register gradient masking hooks ---
        for module in model.modules():
            if isinstance(module, PattentionLayer):
                hook_handles.append(module.key_param_tokens.register_hook(
                    lambda grad, m=module: apply_grad_mask_hook(grad, m.key_grad_mask)
                ))
                hook_handles.append(module.value_param_tokens.register_hook(
                    lambda grad, m=module: apply_grad_mask_hook(grad, m.value_grad_mask)
                ))

        # For non-Pattention parameters, freeze them after the first task
        if task_id > 0:
            hook_handles.append(model.pos_embedding.register_hook(lambda grad: grad * 0))
            hook_handles.append(model.cls_token.register_hook(lambda grad: grad * 0))
            for name, param in model.named_parameters():
                if 'transformer' in name and 'norm' in name:
                     hook_handles.append(param.register_hook(lambda grad: grad * 0))
                if 'to_patch_embedding' in name and 'norm' in name:
                     hook_handles.append(param.register_hook(lambda grad: grad * 0))

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        global_step = train_task(model, task_id, train_loaders[task_id], optimizer, criterion, DEVICE, EPOCHS_PER_TASK, CLASSES_PER_TASK, global_step)
        
        print(f"--- Finished Training on Task {task_id} ---")
        evaluate(model, test_loaders, DEVICE, task_id + 1, CLASSES_PER_TASK)

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on All Tasks ---")
    final_accuracies = evaluate(model, test_loaders, DEVICE, NUM_TASKS, CLASSES_PER_TASK)
    print(f"\nFinal model parameters: {count_parameters(model):,}")
    print(f"Average Accuracy: {np.mean(final_accuracies):.2f}%")

    if WANDB_AVAILABLE:
        wandb.finish()