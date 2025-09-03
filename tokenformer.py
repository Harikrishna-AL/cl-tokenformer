# tokenformer_inference.py
import torch
import torch.nn as nn
from tokenformer import ContinualLearner, PattentionLayer # Import from modified file
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
from copy import deepcopy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging.")

# --- HELPER & UTILITY FUNCTIONS (Unchanged from previous versions) ---
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
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))
        train_indices = [i for i, label in enumerate(full_train_dataset.targets) if label in task_classes]
        test_indices = [i for i, label in enumerate(full_test_dataset.targets) if label in task_classes]
        train_subset, test_subset = Subset(full_train_dataset, train_indices), Subset(full_test_dataset, test_indices)
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
        print(f"Task {task_id}: Classes {task_classes}, Train {len(train_subset)}, Test {len(test_subset)}")
    return train_loaders, test_loaders

def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            start_class = task_id * classes_per_task
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                output = model(data, task_id=num_tasks_seen - 1, training=False)
                # We need to map the target to the output space of the correct head
                task_specific_output = output[:, start_class:start_class + classes_per_task]
                _, predicted = torch.max(task_specific_output.data, 1)
                total += target.size(0)
                correct += (predicted == (target - start_class)).sum().item()
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
    print(f"Evaluation after Task {num_tasks_seen-1}: Accuracies = {['%.2f' % acc for acc in accuracies]}")
    return accuracies

# --- NEW: GRADIENT MASKING FOR FREEZING ---
def apply_grad_mask_hook(grad, mask):
    return grad * mask.unsqueeze(-1)

def apply_masks_and_hooks(model):
    handles = []
    for module in model.modules():
        if isinstance(module, PattentionLayer):
            # Agnostic params are always trainable, no hook needed
            # Specific params need a hook with their corresponding mask
            if module.key_param_specific.requires_grad and module.key_param_specific.numel() > 0:
                handles.append(module.key_param_specific.register_hook(
                    lambda grad, m=module: apply_grad_mask_hook(grad, m.specific_grad_mask)
                ))
                handles.append(module.value_param_specific.register_hook(
                    lambda grad, m=module: apply_grad_mask_hook(grad, m.specific_grad_mask)
                ))
    return handles

# --- NEW: CONTINUAL LEARNING LOSS FUNCTIONS ---
def self_distillation_loss_fn(model_current, model_old, data):
    with torch.no_grad():
        features_old = model_old.get_features(data)
    features_current = model_current.get_features(data)
    return F.mse_loss(features_current, features_old)

def mahalanobis_distance_sq(x, y, inv_cov):
    delta = x - y
    return torch.einsum('bi,ij,bj->b', delta, inv_cov, delta)

def covariance_calibration_loss_fn(model_current, model_old, data, stored_covariances, device):
    loss = 0.0
    num_classes = 0
    with torch.no_grad():
        features_old = model_old.get_features(data)
    features_current = model_current.get_features(data)

    for class_cov in stored_covariances.values():
        inv_cov = torch.inverse(class_cov.to(device))
        # Use pairs of samples to calculate distance
        dist_old = mahalanobis_distance_sq(features_old[:-1], features_old[1:], inv_cov)
        dist_current = mahalanobis_distance_sq(features_current[:-1], features_current[1:], inv_cov)
        loss += F.l1_loss(dist_current, dist_old)
        num_classes += 1
        
    return loss / num_classes if num_classes > 0 else 0.0

# --- NEW: POST-TRAINING CALIBRATION & ALIGNMENT ---
def update_statistics(model, data_loader, device):
    """Calculates and returns the mean and covariance for the current task's data."""
    model.eval()
    all_features = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            features = model.get_features(data)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    mean = torch.mean(all_features, dim=0)
    cov = torch.cov(all_features.T)
    return mean, cov

def mean_shift_compensation(model_current, model_old, data_loader, stored_means, device):
    print("🔧 Calibrating means (Mean Shift Compensation)...")
    model_current.eval()
    model_old.eval()
    
    # Estimate shift using new task's data
    all_shifts = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            features_old = model_old.get_features(data)
            features_current = model_current.get_features(data)
            all_shifts.append((features_current - features_old).cpu())
            
    avg_shift = torch.mean(torch.cat(all_shifts, dim=0), dim=0)
    
    # Apply shift to all old class means
    for task_id in stored_means:
        stored_means[task_id] += avg_shift
        
    print("Mean calibration complete.")
    return stored_means

def classifier_alignment(model, stored_means, stored_covariances, num_tasks_seen, classes_per_task, device, config):
    print("🔧 Aligning classifier...")
    model.train() # Set to train mode to update classifier weights
    
    # Isolate classifier parameters
    classifier_params = []
    for i in range(num_tasks_seen):
        classifier_params.extend(model.mlp_heads[i].parameters())
    
    optimizer = torch.optim.Adam(classifier_params, lr=config["lr_align"])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["align_epochs"]):
        # Generate synthetic features
        features, labels = [], []
        for class_id in range(num_tasks_seen * classes_per_task):
            task_id = class_id // classes_per_task
            mean = stored_means[task_id]
            cov = stored_covariances[task_id]
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            
            # Sample features and create labels
            synth_features = dist.sample((config["align_batch_size"],))
            synth_labels = torch.full((config["align_batch_size"],), fill_value=class_id, dtype=torch.long)
            
            features.append(synth_features)
            labels.append(synth_labels)

        features = torch.cat(features, dim=0).to(device)
        labels = torch.cat(labels, dim=0).to(device)
        
        # Train the classifier head
        optimizer.zero_grad()
        # Pass synthetic features through the corresponding heads
        outputs = []
        for i in range(num_tasks_seen):
            start_idx = i * classes_per_task * config["align_batch_size"]
            end_idx = start_idx + classes_per_task * config["align_batch_size"]
            if end_idx > start_idx:
                 outputs.append(model.mlp_heads[i](features[start_idx:end_idx]))

        output = torch.cat(outputs, dim=0)
        
        # Adjust labels to be task-local
        local_labels = labels % classes_per_task
        
        loss = criterion(output, local_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Classifier Alignment Epoch {epoch+1}/{config['align_epochs']}, Loss: {loss.item():.4f}")
    
    print("Classifier alignment complete.")


# --- MAIN TRAINING SCRIPT ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenformer Continual Learning')
    args = parser.parse_args()

    config = {
        "num_tasks": 5, "classes_per_task": 2, "batch_size": 64,
        "patience": 3, "min_delta_loss": 0.01,
        "lr": 1e-4, "lr_align": 1e-5,
        "num_agnostic_tokens": 64, "num_specific_tokens_per_task": 64,
        "lambda_distill": 1.0, "lambda_cov": 0.5,
        "align_epochs": 20, "align_batch_size": 32, # Samples per class
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if WANDB_AVAILABLE:
        wandb.init(project="tokenformer-cl-calibration", config=config)

    model = ContinualLearner(
        dim=256, depth=4, heads=4, mlp_dim=512,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        num_agnostic_tokens=config["num_agnostic_tokens"], device=DEVICE
    ).to(DEVICE)
    
    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    
    # State tracking
    results_history = {}
    stored_means = {}
    stored_covariances = {}
    model_old = None

    print(f"Initial Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    for task_id in range(config["num_tasks"]):
        print(f"\n{'='*20} Task {task_id} {'='*20}")
        
        if task_id > 0:
            model.grow(config["num_specific_tokens_per_task"])
            print(f"Trainable parameters after growth: {count_parameters(model, trainable_only=True):,}")
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
        
        # --- Main Training Phase ---
        model.train()
        hooks = apply_masks_and_hooks(model)
        
        patience_counter, best_loss, epoch = 0, float('inf'), 0
        while patience_counter < config["patience"]:
            loop = tqdm(train_loaders[task_id], leave=True, desc=f"Task {task_id} | Epoch {epoch+1}")
            epoch_loss = 0.0
            
            for data, target in loop:
                data, target = data.to(device), target.to(device)
                target = target - task_id * config["classes_per_task"]
                optimizer.zero_grad()
                
                # --- Loss Calculation ---
                output = model(data, task_id, training=True)
                class_loss = criterion(output, target)
                total_loss = class_loss
                
                loss_distill, loss_cov = 0.0, 0.0
                if model_old is not None:
                    loss_distill = self_distillation_loss_fn(model, model_old, data)
                    loss_cov = covariance_calibration_loss_fn(model, model_old, data, stored_covariances, DEVICE)
                    total_loss += config["lambda_distill"] * loss_distill
                    total_loss += config["lambda_cov"] * loss_cov

                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                loop.set_postfix(loss=total_loss.item(), cls=class_loss.item(),
                                 dist=f"{loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0:.3f}",
                                 cov=f"{loss_cov.item() if isinstance(loss_cov, torch.Tensor) else 0:.3f}")
            
            avg_epoch_loss = epoch_loss / len(train_loaders[task_id])
            if avg_epoch_loss < best_loss - config["min_delta_loss"]:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            epoch += 1
        
        for handle in hooks: handle.remove() # Clean up hooks
        print(f"Training for Task {task_id} finished after {epoch} epochs.")
        
        # --- Update & Calibrate ---
        mean, cov = update_statistics(model, train_loaders[task_id], DEVICE)
        stored_means[task_id], stored_covariances[task_id] = mean, cov
        
        if model_old is not None:
            stored_means = mean_shift_compensation(model, model_old, train_loaders[task_id], stored_means, DEVICE)
        
        classifier_alignment(model, stored_means, stored_covariances, task_id + 1, config["classes_per_task"], DEVICE, config)

        # --- Evaluate and Prepare for Next Task ---
        accuracies = evaluate(model, test_loaders, DEVICE, task_id + 1, config["classes_per_task"])
        results_history[task_id] = accuracies
        
        model_old = deepcopy(model).to(DEVICE).eval()

        if WANDB_AVAILABLE:
            log_data = {f"task_{i}_acc": acc for i, acc in enumerate(accuracies)}
            log_data["avg_acc"] = np.mean(accuracies)
            log_data["task_id"] = task_id
            wandb.log(log_data)

    if WANDB_AVAILABLE: wandb.finish()