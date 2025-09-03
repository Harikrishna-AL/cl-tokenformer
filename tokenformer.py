# tokenformer_inference.py
import torch
import torch.nn as nn
from vit_pytorch import ContinualLearner, PattentionLayer 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
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

# --- MODIFIED: DATASET WRAPPER FOR CONTRASTIVE LEARNING ---
class ContrastiveWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return (view1, view2), label

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size, image_size):
    # This transform is for creating two augmented views for contrastive learning
    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # A simpler transform for the test set
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    raw_train_dataset = MNIST(root='./data', train=True, download=True)
    # Wrap the training dataset for contrastive learning
    full_train_dataset = ContrastiveWrapper(raw_train_dataset, contrastive_transform)
    full_test_dataset = MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))
        
        train_indices = [i for i, label in enumerate(raw_train_dataset.targets) if label in task_classes]
        test_indices = [i for i, label in enumerate(full_test_dataset.targets) if label in task_classes]
        
        train_subset = Subset(full_train_dataset, train_indices)
        test_subset = Subset(full_test_dataset, test_indices)
        
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
    return train_loaders, test_loaders

# --- NEW: InfoNCE Loss function as per the TagFex paper ---
def info_nce_loss(features1, features2, temperature=0.1, device='cpu'):
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    
    similarity_matrix = torch.matmul(features1, features2.T) / temperature
    
    labels = torch.arange(len(features1), device=device)
    
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
    
# --- NEW: Continual Self-Supervised Training Phase (based on TagFex/CaSSLe) ---
def train_cssl_phase(model, model_old, data_loader, device, config, task_id):
    print(f"\n🔬 Starting Continual SSL Phase for Task {task_id}...")
    model.train()
    # Optimizer for only the agnostic and SSL-related parameters
    optimizer_cssl = torch.optim.Adam(model.ssl_parameters(), lr=config["lr_cssl"])
    
    for epoch in range(config["cssl_epochs"]):
        loop = tqdm(data_loader, leave=False, desc=f"CSSL Epoch {epoch+1}/{config['cssl_epochs']}")
        total_cssl_loss = 0
        
        for (view1, view2), _ in loop:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer_cssl.zero_grad()
            
            # --- Standard Contrastive Loss (SimCLR part) ---
            features1 = model.get_features(view1)
            features2 = model.get_features(view2)
            
            z1 = model.contrastive_projector(features1)
            z2 = model.contrastive_projector(features2)
            
            loss_simclr = info_nce_loss(z1, z2, device=device)
            total_loss = loss_simclr
            
            # --- Predictive Loss (CaSSLe part for t > 0) ---
            loss_cassle = torch.tensor(0.0)
            if model_old is not None:
                p1 = model.contrastive_predictor(z1)
                
                with torch.no_grad():
                    features_old1 = model_old.get_features(view1)
                    z_old1 = model_old.contrastive_projector(features_old1)
                
                loss_cassle = info_nce_loss(p1, z_old1.detach(), device=device)
                total_loss += loss_cassle

            total_loss.backward()
            optimizer_cssl.step()
            
            total_cssl_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), simclr=loss_simclr.item(), cassle=loss_cassle.item())
            
    print(f"🔬 CSSL Phase Complete. Final Avg Loss: {total_cssl_loss / len(data_loader):.4f}")

# --- (Other functions like evaluate, apply_hooks, calibration etc. remain unchanged) ---
def count_parameters(model, trainable_only=False):
    if trainable_only: return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    model.eval(); accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            start_class = task_id * classes_per_task
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                output = model(data, task_id=num_tasks_seen - 1, training=False)
                task_specific_output = output[:, start_class:start_class + classes_per_task]
                _, predicted = torch.max(task_specific_output.data, 1)
                total += target.size(0)
                correct += (predicted == (target - start_class)).sum().item()
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
    print(f"Evaluation after Task {num_tasks_seen-1}: Accuracies = {['%.2f' % acc for acc in accuracies]}")
    return accuracies
def apply_grad_mask_hook(grad, mask): return grad * mask
def apply_masks_and_hooks(model):
    handles = []
    for module in model.modules():
        if isinstance(module, PattentionLayer):
            if module.key_param_specific.requires_grad and module.key_param_specific.numel() > 0:
                handles.append(module.key_param_specific.register_hook(lambda grad, m=module: apply_grad_mask_hook(grad, m.specific_grad_mask)))
                handles.append(module.value_param_specific.register_hook(lambda grad, m=module: apply_grad_mask_hook(grad, m.specific_grad_mask)))
    return handles
def self_distillation_loss_fn(model_current, model_old, data):
    with torch.no_grad(): features_old = model_old.get_features(data)
    features_current = model_current.get_features(data)
    return F.mse_loss(features_current, features_old)
def mahalanobis_distance_sq(x, y, inv_cov):
    delta = x - y
    return torch.einsum('bi,ij,bj->b', delta, inv_cov, delta)
def covariance_calibration_loss_fn(model_current, model_old, data, stored_covariances, device):
    loss = 0.0; num_classes = 0
    with torch.no_grad(): features_old = model_old.get_features(data)
    features_current = model_current.get_features(data)
    for class_cov in stored_covariances.values():
        inv_cov = torch.inverse(class_cov.to(device))
        dist_old = mahalanobis_distance_sq(features_old[:-1], features_old[1:], inv_cov)
        dist_current = mahalanobis_distance_sq(features_current[:-1], features_current[1:], inv_cov)
        loss += F.l1_loss(dist_current, dist_old)
        num_classes += 1
    return loss / num_classes if num_classes > 0 else 0.0
def update_statistics(model, data_loader, device):
    # ... (function is correct, no changes needed) ...
    model.eval()
    all_features = []
    # Note: this function is called with the test_loader, which is NOT contrastive, so it's fine.
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            features = model.get_features(data)
            all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0)
    mean, cov = torch.mean(all_features, dim=0), torch.cov(all_features.T)
    return mean, cov

# --- FUNCTION WITH CORRECTION ---
def mean_shift_compensation(model_current, model_old, data_loader, stored_means, device):
    print("🔧 Calibrating means (Mean Shift Compensation)...")
    model_current.eval()
    model_old.eval()
    all_shifts = []
    with torch.no_grad():
        # --- FIX: Correctly unpack the (view1, view2) tuple from the contrastive data loader ---
        for (view1, _), _ in data_loader:
            # We only need one of the views to estimate the shift
            data = view1.to(device)
            # --- END FIX ---
            
            features_old = model_old.get_features(data)
            features_current = model_current.get_features(data)
            all_shifts.append((features_current - features_old).cpu())
            
    avg_shift = torch.mean(torch.cat(all_shifts, dim=0), dim=0)
    
    for task_id in stored_means:
        stored_means[task_id] += avg_shift
        
    print("Mean calibration complete.")
    return stored_means
    
def classifier_alignment(model, stored_means, stored_covariances, num_tasks_seen, classes_per_task, device, config):
    print("🔧 Aligning classifier...")
    model.train()
    classifier_params = [p for i in range(num_tasks_seen) for p in model.mlp_heads[i].parameters()]
    optimizer = torch.optim.Adam(classifier_params, lr=config["lr_align"])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config["align_epochs"]):
        features, labels = [], []
        for task_id in range(num_tasks_seen):
            mean, cov = stored_means[task_id], stored_covariances[task_id]
            jitter = torch.eye(cov.shape[0], device=device) * 1e-4
            cov_reg = cov.to(device) + jitter
            dist = torch.distributions.MultivariateNormal(mean.to(device), covariance_matrix=cov_reg)
            synth_features = dist.sample((config["align_batch_size"] * classes_per_task,))
            start_class = task_id * classes_per_task
            synth_labels = torch.arange(start_class, start_class + classes_per_task, device=device).repeat_interleave(config["align_batch_size"])
            features.append(synth_features); labels.append(synth_labels)
        features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)
        optimizer.zero_grad()
        output = model(features, task_id=num_tasks_seen - 1, training=False)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0: print(f"  Classifier Alignment Epoch {epoch+1}/{config['align_epochs']}, Loss: {loss.item():.4f}")
    print("Classifier alignment complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenformer Continual Learning'); args = parser.parse_args()
    config = {
        "num_tasks": 5, "classes_per_task": 2, "batch_size": 128, "patience": 3,
        "lr": 1e-4, "lr_cssl": 3e-4, "lr_align": 1e-5, "min_delta_loss": 0.01,
        "num_agnostic_tokens": 64, "num_specific_tokens_per_task": 32,
        "lambda_distill": 0.25, "lambda_cov": 0.005,
        "cssl_epochs": 5, "align_epochs": 100, "align_batch_size": 32,
        "image_size": 32, "patch_size": 4, # Smaller image/patch for faster training
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if WANDB_AVAILABLE: wandb.init(project="tokenformer-cl-cssl", config=config)

    model = ContinualLearner(
        image_size=config["image_size"], patch_size=config["patch_size"],
        dim=256, depth=4, heads=4, mlp_dim=512,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        num_agnostic_tokens=config["num_agnostic_tokens"], device=DEVICE
    ).to(DEVICE)
    
    train_loaders, test_loaders = get_split_mnist_loaders(
        config["num_tasks"], config["classes_per_task"], config["batch_size"], config["image_size"]
    )
    criterion = nn.CrossEntropyLoss()
    
    results_history, stored_means, stored_covariances = {}, {}, {}
    model_old = None
    print(f"Initial Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    for task_id in range(config["num_tasks"]):
        print(f"\n{'='*20} Task {task_id} {'='*20}")
        
        # --- MODIFIED: Run CSSL phase BEFORE supervised training ---
        train_cssl_phase(model, model_old, train_loaders[task_id], DEVICE, config, task_id)

        if task_id > 0:
            model.grow(config["num_specific_tokens_per_task"])
            print(f"Trainable parameters after growth: {count_parameters(model, trainable_only=True):,}")
        
        optimizer_sup = torch.optim.Adam(model.supervised_parameters(), lr=config["lr"])
        
        model.train()
        hooks = apply_masks_and_hooks(model)
        
        patience_counter, best_loss, epoch = 0, float('inf'), 0
        while patience_counter < config["patience"]:
            loop = tqdm(train_loaders[task_id], leave=True, desc=f"Supervised Task {task_id} | Epoch {epoch+1}")
            for (view1, view2), target in loop: # Data loader now yields two views
                data = view1.to(DEVICE) # Use the first view for supervised learning
                target = target.to(DEVICE)
                target = target - task_id * config["classes_per_task"]
                optimizer_sup.zero_grad()
                
                output = model(data, task_id, training=True)
                class_loss = criterion(output, target)
                
                total_loss = class_loss
                loss_distill, loss_cov = 0.0, 0.0
                if model_old is not None:
                    loss_distill = self_distillation_loss_fn(model, model_old, data)
                    loss_cov = covariance_calibration_loss_fn(model, model_old, data, stored_covariances, DEVICE)
                    total_loss = class_loss + config["lambda_distill"] * loss_distill + config["lambda_cov"] * loss_cov

                total_loss.backward()
                optimizer_sup.step()
                loop.set_postfix(loss=total_loss.item(), cls=class_loss.item())
            
            if class_loss.item() < best_loss - config["min_delta_loss"]:
                best_loss = class_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            epoch += 1
        
        for handle in hooks: handle.remove()
        print(f"Supervised training for Task {task_id} finished after {epoch} epochs.")
        
        # For stats, use the test loader for a more representative distribution
        mean, cov = update_statistics(model, test_loaders[task_id], DEVICE)
        stored_means[task_id], stored_covariances[task_id] = mean, cov
        
        if model_old is not None:
            # Use the current train loader for MSC as it contains the new data distribution
            stored_means = mean_shift_compensation(model, model_old, train_loaders[task_id], stored_means, DEVICE)
        
        classifier_alignment(model, stored_means, stored_covariances, task_id + 1, config["classes_per_task"], DEVICE, config)

        accuracies = evaluate(model, test_loaders, DEVICE, task_id + 1, config["classes_per_task"])
        results_history[task_id] = accuracies
        
        model_old = deepcopy(model).to(DEVICE).eval()

        if WANDB_AVAILABLE:
            log_data = {f"task_{i}_acc": acc for i, acc in enumerate(accuracies)}
            log_data["avg_acc"] = np.mean(accuracies)
            log_data["task_id"] = task_id
            wandb.log(log_data)

    if WANDB_AVAILABLE: wandb.finish()