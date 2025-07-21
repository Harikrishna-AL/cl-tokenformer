import torch
import torch.nn as nn
from vit_pytorch import ViT, TokenformerViT
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np


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

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    """ Prepares the Split MNIST dataloaders. """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loaders = []
    test_loaders = []

    for task_id in range(num_tasks):
        # Determine the classes for the current task
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))

        # Filter dataset for task-specific classes
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label in task_classes]
        
        # Create subsets
        train_subset = Subset(full_train_dataset, train_indices)
        test_subset = Subset(full_test_dataset, test_indices)

        # Create dataloaders
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False))
        
        print(f"Task {task_id}: Classes {task_classes}, Train samples {len(train_subset)}, Test samples {len(test_subset)}")

    return train_loaders, test_loaders

def train_task(model, task_id, train_loader, optimizer, criterion, device, epochs):
    """ Trains the model on a single task. """
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            
            # Adjust target labels to be 0-indexed for the current task
            target = target % len(train_loader.dataset.dataset.classes[task_id*2:(task_id+1)*2])

            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Task {task_id} Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())

def evaluate(model, test_loaders, device, num_tasks):
    """ Evaluates the model on all seen tasks. """
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks):
            correct = 0
            total = 0
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                target = target % len(test_loaders[task_id].dataset.dataset.classes[task_id*2:(task_id+1)*2])
                
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
    print(f"Device: {DEVICE}")

    # --- Model Configuration ---
    model = TokenformerViT(
        image_size=28,
        patch_size=7,
        num_tasks=NUM_TASKS,
        classes_per_task=CLASSES_PER_TASK,
        channels=1,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        device=DEVICE
    ).to(DEVICE)

    # --- Data Preparation ---
    train_loaders, test_loaders = get_split_mnist_loaders(NUM_TASKS, CLASSES_PER_TASK, BATCH_SIZE)
    
    criterion = nn.CrossEntropyLoss()

    # --- Main Continual Learning Loop ---
    trained_params = []
    for task_id in range(NUM_TASKS):
        print(f"\n--- Starting Training on Task {task_id} ---")
        
        # Freeze all previously trained parameters
        # for param in model.parameters():
        #     param.requires_grad = False
            
        # # Unfreeze the parameters for the current task's head
        # for param in model.mlp_heads[task_id].parameters():
        #     param.requires_grad = True

        # # Unfreeze the core model parameters ONLY for the first task
        # if task_id == 0:
        #     for module in [model.to_patch_embedding, model.transformer]:
        #          for param in module.parameters():
        #             param.requires_grad = True
        #     model.cls_token.requires_grad = True
        #     model.pos_embedding.requires_grad = True
        
        # For subsequent tasks, only the new parts of grown Pattention layers are trainable
        # (This requires more complex parameter management, for now we train the head only)
        # A simple approach for this demo: only train the new head after task 0.
        # A more advanced approach would track new parameter tokens and only set them to requires_grad=True
        if task_id == 0:
            model.pos_embedding.requires_grad = True
            model.cls_token.requires_grad = True
            # The Pattention layers are trainable by default on task 0
            for param in model.to_patch_embedding.parameters(): param.requires_grad = True
            for param in model.transformer.parameters(): param.requires_grad = True
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=LR)

        train_task(model, task_id, train_loaders[task_id], optimizer, criterion, DEVICE, EPOCHS_PER_TASK)
        
        print(f"--- Finished Training on Task {task_id} ---")
        evaluate(model, test_loaders, DEVICE, task_id + 1)
        print(f"Num model parameters: {count_parameters(model)}")
        print(f'Percentage of frozen params: {calculate_frozen_percentage(model)}')
        
        # For this simplified demo, we won't grow the model to keep it runnable.
        # To implement the full paper's idea, you would call:
        if task_id < NUM_TASKS - 1:
            for param in model.parameters():
                param.requires_grad = False
            model.grow() 
        # And then manage which new parameters are trainable.

        for param in model.mlp_heads[task_id].parameters():
            param.requires_grad = True

         # For the first task, also train the non-Pattention backbone components
        

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on All Tasks ---")
    final_accuracies = evaluate(model, test_loaders, DEVICE, NUM_TASKS)
    print(f"\nAverage Accuracy: {np.mean(final_accuracies):.2f}%")
