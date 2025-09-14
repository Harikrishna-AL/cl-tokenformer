# main_ipca.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse

# Import from your other files
from vit_pytorch import ContinualLearner, PattentionLayer # CORRECTED IMPORT
from ipca import IncrementalPCA

# --- Helper Functions (from previous script) ---
def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))
        train_indices = [i for i, t in enumerate(full_train_dataset.targets) if t in task_classes]
        test_indices = [i for i, t in enumerate(full_test_dataset.targets) if t in task_classes]
        train_loaders.append(DataLoader(Subset(full_train_dataset, train_indices), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(full_test_dataset, test_indices), batch_size=batch_size, shuffle=False))
        print(f"Task {task_id}: Classes {task_classes}, Train {len(train_indices)}, Test {len(test_indices)}")
    return train_loaders, test_loaders

# --- Gradient Masking Logic ---
def apply_grad_mask_hook(grad, mask):
    """Multiplies gradient by a mask element-wise."""
    return grad * mask

def apply_masks_and_hooks(model, previous_handles):
    """
    Removes old hooks and applies new ones based on the `grad_mask` buffers
    in each PattentionLayer. This freezes the parameters for old tasks.
    """
    for handle in previous_handles:
        handle.remove()
    
    new_handles = []
    for module in model.growing_transformer.modules():
        if isinstance(module, PattentionLayer):
            if module.key_param_tokens.grad is not None:
                module.key_param_tokens.grad.zero_()
            new_handles.append(
                module.key_param_tokens.register_hook(
                    lambda grad, m=module: apply_grad_mask_hook(grad, m.key_grad_mask)
                )
            )
            
            if module.value_param_tokens.grad is not None:
                module.value_param_tokens.grad.zero_()
            new_handles.append(
                module.value_param_tokens.register_hook(
                    lambda grad, m=module: apply_grad_mask_hook(grad, m.value_grad_mask)
                )
            )
    return new_handles

# --- Phase 1: Train Backbone and Learn Distributions ---
def train_phase_1(config, device):
    print("--- Starting Phase 1: Training Backbone and Learning Class Distributions ---")
    
    # Setup model, data, and optimizer
    model = ContinualLearner(
        image_size=config['image_size'], dim=config['dim'], depth=config['depth'],
        heads=config['heads'], mlp_dim=config['mlp_dim'], num_tasks=config['num_tasks'],
        classes_per_task=config['classes_per_task'], device=device
    ).to(device)
    train_loaders, _ = get_split_mnist_loaders(config['num_tasks'], config['classes_per_task'], config['batch_size'])
    criterion = nn.CrossEntropyLoss()
    
    ipca_models = {} # Dictionary to store one IncrementalPCA model per class
    hook_handles = [] # To store hook handles

    for task_id in range(config['num_tasks']):
        print(f"\nTraining on Task {task_id}...")
        if task_id > 0:
            model.grow()
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
        hook_handles = apply_masks_and_hooks(model, hook_handles)

        model.train()
        for epoch in range(config['epochs_per_task']):
            epoch_loss = 0.0
            loop = tqdm(train_loaders[task_id], desc=f"Task {task_id} Epoch {epoch+1}", leave=False)
            for data, target in loop:
                data, target = data.to(device), target.to(device)
                target_local = target - task_id * config['classes_per_task']
                
                optimizer.zero_grad()
                output = model(data, task_id, training=True)
                loss = criterion(output, target_local)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            avg_epoch_loss = epoch_loss / len(train_loaders[task_id])
            print(f"Task {task_id} Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        
        print(f"Finished training on Task {task_id}. Now fitting distributions...")
        
        model.eval()
        with torch.no_grad():
            all_features, all_labels = [], []
            for data, target in train_loaders[task_id]:
                data = data.to(device)
                features = model.forward_features(data)
                all_features.append(features.cpu())
                all_labels.append(target.cpu())
            
            all_features = torch.cat(all_features)
            all_labels = torch.cat(all_labels)

            start_class = task_id * config['classes_per_task']
            end_class = (task_id + 1) * config['classes_per_task']
            for class_idx in range(start_class, end_class):
                class_features = all_features[all_labels == class_idx]
                if class_idx not in ipca_models:
                    ipca_models[class_idx] = IncrementalPCA(n_components=config['ipca_components'], device='cpu')
                ipca_models[class_idx].update(class_features)
                print(f"Updated iPCA for class {class_idx} with {len(class_features)} samples.")

    for handle in hook_handles:
        handle.remove()

    torch.save(model.state_dict(), config['backbone_path'])
    ipca_states = {k: v.state_dict() for k, v in ipca_models.items()}
    torch.save(ipca_states, config['ipca_path'])
    print(f"\nPhase 1 complete. Backbone saved to {config['backbone_path']}")
    print(f"iPCA models saved to {config['ipca_path']}")


# --- Phase 2: Train Final Classifier on Synthetic Data ---
def train_phase_2(config, device):
    print("\n--- Starting Phase 2: Training Final Classifier on Synthetic Features ---")
    
    ipca_states = torch.load(config['ipca_path'])
    ipca_models = {}
    for class_idx, state_dict in ipca_states.items():
        model = IncrementalPCA(n_components=state_dict['n_components'], device=device)
        model.load_state_dict(state_dict)
        ipca_models[class_idx] = model
    print(f"Loaded {len(ipca_models)} iPCA models.")

    synthetic_features, synthetic_labels = [], []
    for class_idx, ipca_model in ipca_models.items():
        samples = ipca_model.sample(config['samples_per_class'])
        synthetic_features.append(samples)
        synthetic_labels.append(torch.full((config['samples_per_class'],), class_idx, dtype=torch.long, device=device))

    synthetic_dataset = TensorDataset(torch.cat(synthetic_features), torch.cat(synthetic_labels))
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # The final classifier now predicts one of the N tasks
    final_classifier = nn.Linear(config['dim'], config['num_tasks']).to(device)
    optimizer = optim.Adam(final_classifier.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    print("Training final task-level classifier (Router)...")
    final_classifier.train()
    for epoch in range(config['final_classifier_epochs']):
        total_loss = 0
        for features, labels in tqdm(synthetic_loader, desc=f"Phase 2 Epoch {epoch+1}"):
            optimizer.zero_grad()
            features = features.to(device)
            
            # --- CORRECTED: Train with task labels, not class labels ---
            # The classifier predicts a task ID, so the label must be a task ID.
            task_labels = labels // config['classes_per_task']
            
            outputs = final_classifier(features)
            # print(outputs, task_labels)
            
            loss = criterion(outputs, task_labels)
            # --- END CORRECTION ---

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(synthetic_loader):.4f}")
        print(torch.argmax(outputs, dim=1), task_labels)

    torch.save(final_classifier.state_dict(), config['classifier_path'])
    print(f"Phase 2 complete. Final classifier saved to {config['classifier_path']}")

# --- Evaluation ---
def evaluate(config, device):
    print("\n--- Starting Evaluation on Real Test Data ---")
    
    backbone = ContinualLearner(
        image_size=config['image_size'], dim=config['dim'], depth=config['depth'],
        heads=config['heads'], mlp_dim=config['mlp_dim'], num_tasks=config['num_tasks'],
        classes_per_task=config['classes_per_task'], device=device
    ).to(device)
    for _ in range(config['num_tasks'] - 1): backbone.grow()
    backbone.load_state_dict(torch.load(config['backbone_path'], map_location=device))
    backbone.eval()

    final_classifier = nn.Linear(config['dim'], config['num_tasks']).to(device)
    final_classifier.load_state_dict(torch.load(config['classifier_path'], map_location=device))
    final_classifier.eval()

    _, test_loaders = get_split_mnist_loaders(config['num_tasks'], config['classes_per_task'], config['batch_size'])
    full_test_loader = DataLoader(torch.utils.data.ConcatDataset([dl.dataset for dl in test_loaders]), batch_size=config['batch_size'])

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(full_test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            
            # --- COMPLETED: Two-stage prediction logic ---
            # 1. Get features from the backbone
            features = backbone.forward_features(data)
            
            # 2. Use the final classifier as a "Router" to predict the task
            task_logits = final_classifier(features)
            predicted_task_ids = torch.argmax(task_logits, dim=1)
            
            # 3. Use the predicted task to select the correct "Expert" head and get final class prediction
            batch_size = data.shape[0]
            final_predictions = torch.zeros_like(target)
            for i in range(batch_size):
                task_id = predicted_task_ids[i].item()
                feature_sample = features[i].unsqueeze(0) # Add batch dim for the head
                
                # Select the expert head
                expert_head = backbone.mlp_heads[task_id]
                
                # Get local class logits (e.g., for classes {6,7}, logits for {0,1})
                local_logits = expert_head(feature_sample)
                local_prediction = torch.argmax(local_logits, dim=1).item()

                # print(local_prediction, task_id, target[i])
                
                # Convert local prediction to global class ID
                global_prediction = task_id * config['classes_per_task'] + local_prediction
                final_predictions[i] = global_prediction

            total += target.size(0)
            correct += (final_predictions == target).sum().item()
            # --- END COMPLETION ---

    accuracy = 100 * correct / total
    print(f"\nFinal CIL Accuracy on the entire test set: {accuracy:.2f}%")
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenformer with Incremental PCA for CIL')
    parser.add_argument('--phase', required=True, type=str, choices=['1', '2', 'eval', 'all'],
                        help="Which part of the process to run: '1' for backbone training, "
                             "'2' for final classifier training, 'eval' for evaluation, or 'all' to run sequentially.")
    args = parser.parse_args()

    config = {
        # Model HParams
        "image_size": 28, "dim": 128, "depth": 2, "heads": 4, "mlp_dim": 256,
        # CL HParams
        "num_tasks": 5, "classes_per_task": 2,
        # Training HParams
        "batch_size": 128, "lr": 1e-4, "epochs_per_task": 3, "final_classifier_epochs": 20,
        # iPCA HParams
        "ipca_components": 20, "samples_per_class": 1000,
        # File Paths
        "backbone_path": "checkpoints/backbone_final.pth",
        "ipca_path": "checkpoints/ipca_models.pth",
        "classifier_path": "checkpoints/final_classifier.pth",
    }
    
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    os.makedirs("checkpoints", exist_ok=True)

    if args.phase == '1' or args.phase == 'all':
        train_phase_1(config, DEVICE)
    
    if args.phase == '2' or args.phase == 'all':
        train_phase_2(config, DEVICE)
        
    if args.phase == 'eval' or args.phase == 'all':
        evaluate(config, DEVICE)