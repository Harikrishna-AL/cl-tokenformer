import torch
import torch.nn as nn
from vit_pytorch import ContinualLearner # Assuming the edited vit_pytorch.py is in the same directory
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
from einops import rearrange, repeat

# --- Utility Functions (Unchanged) ---
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
# Other utilities like load_checkpoint, print_results_table, etc. remain the same...
def load_checkpoint(model, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=DEVICE)
        start_task_idx, global_step, results_history = checkpoint['current_task_id'] + 1, checkpoint['global_step'], checkpoint['results_history']
        if checkpoint['current_task_id'] > 0:
            print(f"Growing model to saved state (Task {checkpoint['current_task_id']})...")
            for _ in range(checkpoint['current_task_id']): model.grow()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"=> Loaded checkpoint! Resuming from Task {start_task_idx}")
        return model, start_task_idx, global_step, results_history
    else:
        print(f"=> No checkpoint found at '{filename}'"); return model, 0, 0, {}
def print_results_table(history, num_tasks):
    print("\n\n--- Final Results Summary ---")
    header = f"{'After Training Task':<25}" + "".join([f"  Task {i} Acc (%) " for i in range(num_tasks)]) + "  Average Acc (%)"
    print(header + "\n" + "-" * len(header))
    for tid, accs in history.items():
        row = f"{f'Task {tid}':<25}" + "".join([f"    {acc:<10.2f}" for acc in accs]) + "".join([f"    {'--':<10}" for _ in range(num_tasks - len(accs))]) + f"    {np.mean(accs):<10.2f}"
        print(row)
    print("-" * len(header))
def count_parameters(model, trainable_only=False):
    params = model.parameters()
    if trainable_only: params = filter(lambda p: p.requires_grad, params)
    return sum(p.numel() for p in params)

# --- Data Loading (Unchanged) ---
def preprocess_mnist_to_disk(root='./data'):
    preprocessed_dir = os.path.join(root, "mnist_preprocessed")
    if os.path.exists(preprocessed_dir): return
    print(f"⚠️ No preprocessed data found. Creating cache at {preprocessed_dir}...")
    os.makedirs(preprocessed_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(224), transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for split in ['train', 'test']:
        raw_dataset, split_dir = MNIST(root=root, train=(split == 'train'), download=True), os.path.join(preprocessed_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i, (img, label) in enumerate(tqdm(raw_dataset, desc=f"Preprocessing {split} set")):
            torch.save((transform(img), label), os.path.join(split_dir, f"sample_{i}.pt"))
class PreprocessedMNIST(Dataset):
    def __init__(self, root='./data', train=True):
        split = 'train' if train else 'test'
        self.data_dir = os.path.join(root, "mnist_preprocessed", split)
        self.samples = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pt')]
    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return torch.load(self.samples[index])
def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    preprocess_mnist_to_disk()
    train_loaders, test_loaders = [], []
    raw_mnist_train, raw_mnist_test = MNIST(root='./data', train=True, download=True), MNIST(root='./data', train=False, download=True)
    full_train_dataset, full_test_dataset = PreprocessedMNIST(train=True), PreprocessedMNIST(train=False)
    for tid in range(num_tasks):
        task_classes = list(range(tid * classes_per_task, (tid + 1) * classes_per_task))
        train_indices = [i for i, label in enumerate(raw_mnist_train.targets) if label in task_classes]
        test_indices = [i for i, label in enumerate(raw_mnist_test.targets) if label in task_classes]
        train_loaders.append(DataLoader(Subset(full_train_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))
        test_loaders.append(DataLoader(Subset(full_test_dataset, test_indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
    return train_loaders, test_loaders


### --- MODIFIED: BUFFER STORES RESNET FEATURES, SEPARATED FEATURES, AND LABELS --- ###
class CuratedRehearsalBuffer:
    def __init__(self, samples_per_task=10):
        self.samples_per_task = samples_per_task
        self.buffer = {} # Maps task_id to a tuple of (resnet_features, separated_features, labels)

    def add_task_samples(self, task_id, resnet_features, separated_features, losses, labels):
        if resnet_features is None or separated_features is None or losses is None or labels is None:
            return
        _, top_k_indices = torch.topk(losses, k=min(self.samples_per_task, len(losses)))
        
        self.buffer[task_id] = (
            resnet_features[top_k_indices].detach().cpu(),
            separated_features[top_k_indices].detach().cpu(),
            labels[top_k_indices].detach().cpu()
        )
        print(f"✅ Stored {len(top_k_indices)} hardest samples for Task {task_id} in the buffer.")

    def sample(self, batch_size):
        if not self.buffer:
            return None, None, None
        
        all_resnet_feats = torch.cat([item[0] for item in self.buffer.values()], dim=0)
        all_separated_feats = torch.cat([item[1] for item in self.buffer.values()], dim=0)
        all_labels = torch.cat([item[2] for item in self.buffer.values()], dim=0)
        
        if len(all_resnet_feats) == 0:
            return None, None, None

        indices = np.random.choice(len(all_resnet_feats), size=min(batch_size, len(all_resnet_feats)), replace=False)
        return all_resnet_feats[indices], all_separated_feats[indices], all_labels[indices]


def calculate_feature_orthogonality_loss(features):
    if features is None or features.shape[0] < 2: return 0.0
    features_norm = F.normalize(features, p=2, dim=1)
    covariance_matrix = torch.matmul(features_norm.T, features_norm)
    identity = torch.eye(features.shape[1], device=features.device)
    off_diagonal_covariance = covariance_matrix * (1 - identity)
    return torch.mean(off_diagonal_covariance**2)

def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                output, _, _, _ = model(data, task_id=task_id, training=False)
                local_target = target - task_id * classes_per_task
                _, predicted = torch.max(output.data, 1)
                total += local_target.size(0)
                correct += (predicted == local_target).sum().item()
            accuracies.append(100 * correct / total)
    return accuracies


### --- NEW: WAKE AND SLEEP TRAINING FUNCTIONS --- ###

def wake_phase_training(model, current_task_id, train_loader, optimizer_sep_ae, optimizer_head, criterion, rehearsal_buffer, device, classes_per_task, global_step, config):
    print(f"\n🧠 WAKE PHASE: Learning new Task {current_task_id}...")
    model.train()
    
    for param in model.continual_learning_params(): param.requires_grad = False
    for param in model.separation_layer_params(): param.requires_grad = True
    for param in model.mlp_heads[current_task_id].parameters(): param.requires_grad = True

    recon_criterion = nn.MSELoss()
    for epoch in range(config["wake_epochs"]):
        loop = tqdm(train_loader, leave=False, desc=f"WAKE Epoch {epoch+1}/{config['wake_epochs']}")
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            optimizer_sep_ae.zero_grad(); optimizer_head.zero_grad()
            
            output, resnet_feats, separated_feats, recon_feats = model(data, current_task_id)
            
            # Loss 1: Classification for the new task
            class_loss = criterion(output, target - current_task_id * classes_per_task)
            # Loss 2: Reconstruction to preserve information
            recon_loss = recon_criterion(recon_feats, resnet_feats)
            # Loss 3: Orthogonality against past tasks
            ortho_loss = 0.0
            _, past_separated_feats, _ = rehearsal_buffer.sample(data.size(0))
            if past_separated_feats is not None:
                past_separated_feats = past_separated_feats.to(device)
                combined_separated_feats = torch.cat([separated_feats, past_separated_feats], dim=0)
                ortho_loss = calculate_feature_orthogonality_loss(combined_separated_feats)
            
            total_loss = class_loss + config["lambda_recon"] * recon_loss + config["lambda_feat_ortho"] * ortho_loss
            total_loss.backward()
            optimizer_sep_ae.step(); optimizer_head.step()
            global_step += 1
            loop.set_postfix(loss=total_loss.item(), cls=class_loss.item(), rec=recon_loss.item(), ort=ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else 0)

    for param in model.parameters(): param.requires_grad = True # Unfreeze for next phase
        
    model.eval()
    all_resnet_feats, all_separated_feats, all_losses, all_labels = [], [], [], []
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output, resnet_feats, separated_feats, _ = model(data, current_task_id)
            per_sample_loss = nn.CrossEntropyLoss(reduction='none')(output, target - current_task_id * classes_per_task)
            all_resnet_feats.append(resnet_feats.cpu())
            all_separated_feats.append(separated_feats.cpu())
            all_losses.append(per_sample_loss.cpu())
            all_labels.append(target.cpu())
            
    return global_step, torch.cat(all_resnet_feats), torch.cat(all_separated_feats), torch.cat(all_losses), torch.cat(all_labels)


def sleep_phase_consolidation(model, optimizers, criterion, rehearsal_buffer, device, global_step, config):
    print(f"😴 SLEEP PHASE: Consolidating all learned knowledge...")
    if not rehearsal_buffer.buffer:
        print("Buffer is empty, skipping sleep phase."); return global_step
        
    model.train()
    # Freeze the backbone, only train the separation layer and the tokenformer
    for param in model.backbone.parameters(): param.requires_grad = False
    
    for epoch in range(config["sleep_epochs"]):
        resnet_feats, _, labels = rehearsal_buffer.sample(config["buffer_size"])
        if resnet_feats is None: continue
        
        buffer_dataset = TensorDataset(resnet_feats, labels)
        buffer_loader = DataLoader(buffer_dataset, batch_size=config["batch_size"], shuffle=True)
        
        loop = tqdm(buffer_loader, leave=False, desc=f"SLEEP Epoch {epoch+1}/{config['sleep_epochs']}")
        for resnet_f, label_g in loop:
            resnet_f, label_g = resnet_f.to(device), label_g.to(device)
            for opt in optimizers: opt.zero_grad()
            
            # Pass replayed ResNet features through the rest of the model
            separated_f, recon_f = model.separation_autoencoder(resnet_f)
            b, n, _ = separated_f.shape
            cls_tokens = repeat(model.cls_token, '1 1 d -> b 1 d', b=b)
            tokens = torch.cat((cls_tokens, separated_f), dim=1) + model.pos_embedding
            cls_output = model.growing_transformer(tokens)[:, 0]

            # Calculate losses
            ortho_loss = calculate_feature_orthogonality_loss(torch.mean(separated_f, dim=1))
            
            # Classification loss requires routing to the correct head
            class_loss = 0.0
            unique_tasks_in_batch = torch.unique(label_g // config["classes_per_task"])
            for task_id in unique_tasks_in_batch:
                mask = (label_g // config["classes_per_task"] == task_id)
                task_cls_output = cls_output[mask]
                task_labels = label_g[mask]
                
                head_output = model.mlp_heads[task_id](task_cls_output)
                class_loss += criterion(head_output, task_labels - task_id * config["classes_per_task"])

            total_loss = config["lambda_feat_ortho_sleep"] * ortho_loss + class_loss
            total_loss.backward()
            for opt in optimizers: opt.step()
            global_step += 1
            loop.set_postfix(loss=total_loss.item(), ort=ortho_loss.item(), cls=class_loss.item())
            
    return global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenformer Continual Learning'); parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint'); args = parser.parse_args()

    config = {
        "num_tasks": 5, "classes_per_task": 2, "batch_size": 32, "lr": 1e-4,
        "lambda_feat_ortho": 2.0, "lambda_recon": 0.5, "lambda_feat_ortho_sleep": 2.5,
        "samples_per_task": 20, "buffer_size": 200,
        "wake_epochs": 5, "sleep_epochs": 3,
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = ContinualLearner(dim=128, depth=2, heads=4, mlp_dim=256, num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"], device=DEVICE).to(DEVICE)
    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    rehearsal_buffer = CuratedRehearsalBuffer(samples_per_task=config["samples_per_task"])

    start_task_idx, global_step, current_task_id, results_history = 0, 0, 0, {}

    if args.resume: model, start_task_idx, global_step, results_history = load_checkpoint(model, args.resume)
    current_task_id = start_task_idx - 1 if start_task_idx > 0 else 0
    
    print(f"Total model parameters: {count_parameters(model):,}")

    for data_task_idx in range(start_task_idx, config["num_tasks"]):
        print(f"\n{'='*20} Task {data_task_idx} {'='*20}")
        current_task_id = data_task_idx
        if data_task_idx > 0: model.grow()
        
        # Define optimizers for each phase
        optimizer_sep_ae = torch.optim.Adam(model.separation_layer_params(), lr=config["lr"])
        optimizer_head_current = torch.optim.Adam(model.mlp_heads[current_task_id].parameters(), lr=config["lr"])
        optimizer_main = torch.optim.Adam(model.continual_learning_params(), lr=config["lr"])
        
        # --- PHASE 1: WAKE ---
        global_step, res_feats, sep_feats, losses, labels = wake_phase_training(
            model, current_task_id, train_loaders[data_task_idx], optimizer_sep_ae, optimizer_head_current, 
            criterion, rehearsal_buffer, DEVICE, config["classes_per_task"], global_step, config)
        rehearsal_buffer.add_task_samples(current_task_id, res_feats, sep_feats, losses, labels)

        # --- PHASE 2: SLEEP ---
        if data_task_idx > 0:
            global_step = sleep_phase_consolidation(
                model, [optimizer_main, optimizer_sep_ae], criterion, 
                rehearsal_buffer, DEVICE, global_step, config)

        print(f"\n--- Evaluating after Task {data_task_idx} ---")
        accuracies = evaluate(model, test_loaders, DEVICE, current_task_id + 1, config["classes_per_task"])
        avg_accuracy = np.mean(accuracies)
        print(f"--- Average Accuracy across all {len(accuracies)} seen tasks: {avg_accuracy:.2f}% ---")
        
        results_history[current_task_id] = accuracies
        save_checkpoint({
            'current_task_id': current_task_id, 'global_step': global_step,
            'model_state_dict': model.state_dict(), 'results_history': results_history}, 
            filename=f"checkpoint_task_{current_task_id}_final.pth.tar")

    print_results_table(results_history, config["num_tasks"])