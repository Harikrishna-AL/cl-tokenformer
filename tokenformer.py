# tokenformer_inference.py
import torch
import torch.nn as nn
from vit_pytorch import ContinualLearner, PattentionLayer # Assuming vit_pytorch.py is your model file
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Skipping W&B logging. To install: pip install wandb")

# ... (Checkpointing, results table, parameter counting, and data preprocessing functions are unchanged) ...
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        start_task_idx = checkpoint['current_task_id'] + 1
        global_step = checkpoint['global_step']
        results_history = checkpoint['results_history']
        
        if checkpoint['current_task_id'] > 0:
            print(f"Growing model to saved state (Task {checkpoint['current_task_id']})...")
            for _ in range(checkpoint['current_task_id']):
                model.grow()

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"=> Loaded checkpoint! Resuming from Task {start_task_idx}")
        return model, optimizer, start_task_idx, global_step, results_history
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return model, optimizer, 0, 0, {}

def print_results_table(history, num_tasks):
    print("\n\n--- Final Results Summary ---")
    header = f"{'After Training Task':<25}"
    for i in range(num_tasks):
        header += f"  Task {i} Acc (%) "
    header += "  Average Acc (%)"
    print(header)
    print("-" * len(header))
    for trained_task_id, accs in history.items():
        row = f"{f'Task {trained_task_id}':<25}"
        for i in range(len(accs)):
            row += f"    {accs[i]:<10.2f}"
        for i in range(num_tasks - len(accs)):
            row += f"    {'--':<10}"
        avg_acc = np.mean(accs)
        row += f"    {avg_acc:<10.2f}"
        print(row)
    print("-" * len(header))

def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def preprocess_mnist_to_disk(root='./data'):
    preprocessed_dir = os.path.join(root, "mnist_preprocessed")
    if os.path.exists(preprocessed_dir):
        print(f"✔️ Preprocessed data found at {preprocessed_dir}")
        return

    print(f"⚠️ No preprocessed data found. Creating cache at {preprocessed_dir}...")
    os.makedirs(preprocessed_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(224), transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for split in ['train', 'test']:
        is_train = (split == 'train')
        raw_dataset = MNIST(root=root, train=is_train, download=True)
        split_dir = os.path.join(preprocessed_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i, (img, label) in enumerate(tqdm(raw_dataset, desc=f"Preprocessing {split} set")):
            transformed_img = transform(img)
            save_path = os.path.join(split_dir, f"sample_{i}.pt")
            torch.save((transformed_img, label), save_path)

class PreprocessedMNIST(Dataset):
    def __init__(self, root='./data', train=True):
        split = 'train' if train else 'test'
        self.data_dir = os.path.join(root, "mnist_preprocessed", split)
        self.samples = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pt')]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return torch.load(self.samples[index])

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size):
    preprocess_mnist_to_disk()
    full_train_dataset = PreprocessedMNIST(train=True)
    full_test_dataset = PreprocessedMNIST(train=False)
    raw_mnist_train = MNIST(root='./data', train=True, download=True)
    raw_mnist_test = MNIST(root='./data', train=False, download=True)
    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start_class, end_class))
        train_indices = [i for i, label in enumerate(raw_mnist_train.targets) if label in task_classes]
        test_indices = [i for i, label in enumerate(raw_mnist_test.targets) if label in task_classes]
        train_subset, test_subset = Subset(full_train_dataset, train_indices), Subset(full_test_dataset, test_indices)
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))
        test_loaders.append(DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
        print(f"Task {task_id}: Classes {task_classes}, Train samples {len(train_subset)}, Test samples {len(test_subset)}")
    return train_loaders, test_loaders

def calculate_orthogonality_loss(growing_module):
    ortho_loss = 0.0
    num_pairs = 0
    for module in growing_module.modules():
        if isinstance(module, PattentionLayer) and len(module.growth_indices) > 0:
            boundaries = [0] + module.growth_indices + [module.key_param_tokens.shape[0]]
            for i in range(len(boundaries) - 1):
                for j in range(i + 1, len(boundaries) - 1):
                    v_i = module.value_param_tokens[boundaries[i]:boundaries[i+1]]
                    v_j = module.value_param_tokens[boundaries[j]:boundaries[j+1]]
                    v_i_norm = F.normalize(v_i, p=2, dim=1)
                    v_j_norm = F.normalize(v_j, p=2, dim=1)
                    cosine_sim_matrix = torch.matmul(v_i_norm, v_j_norm.T)
                    ortho_loss += torch.mean(cosine_sim_matrix**2)
                    num_pairs += 1
    return ortho_loss / num_pairs if num_pairs > 0 else 0.0

def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    model.eval()
    end_to_end_accuracies = []
    
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            total, router_correct, end_to_end_correct = 0, 0, 0
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                all_logits, cls_output = model(data, task_id=task_id, training=False)
                
                router_logits = model.task_router(cls_output)
                predicted_task_ids = torch.argmax(router_logits, dim=1)
                router_correct += (predicted_task_ids == task_id).sum().item()
                
                _, predicted_global_class = torch.max(all_logits.data, 1)
                end_to_end_correct += (predicted_global_class == target).sum().item()
                total += target.size(0)

            router_acc = 100 * router_correct / total
            end_to_end_acc = 100 * end_to_end_correct / total
            end_to_end_accuracies.append(end_to_end_acc)
            print(f"Accuracy on Task {task_id}: Router Acc: {router_acc:.2f}%, E2E Class Acc: {end_to_end_acc:.2f}%")
            
    return end_to_end_accuracies

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

### --- DELETED: CuratedRehearsalBuffer class, separation_loss_fn, and sleep_phase_consolidation --- ###

### --- MODIFIED: Streamlined training loop --- ###
def train_until_plateau(model, current_task_id, train_loader, optimizer, criterion, device,
                        classes_per_task, global_step, config):
    model.train()
    hook_handles = apply_masks_and_hooks(model, current_task_id, [])
    patience, min_delta = config["patience"], config["min_delta_loss"]
    lambda_ortho, lambda_router = config["lambda_ortho"], config["lambda_router"]
    
    patience_counter, epoch = 0, 0
    best_loss = float('inf')

    print(f"🚀 Starting training for model task {current_task_id} (patience={patience}, lambda_ortho={lambda_ortho}, lambda_router={lambda_router}).")
    
    while patience_counter < patience:
        loop = tqdm(train_loader, leave=True)
        epoch_loss, num_batches = 0.0, 0

        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            target_in_task = target - current_task_id * classes_per_task
            
            optimizer.zero_grad(set_to_none=True)
            
            classification_logits, router_logits = model(data, current_task_id, training=True)
            
            task_loss = criterion(classification_logits, target_in_task)
            
            ortho_loss = 0.0
            if current_task_id > 0:
                ortho_loss = calculate_orthogonality_loss(model.growing_transformer)
            
            router_target = torch.full((data.size(0),), current_task_id, dtype=torch.long, device=device)
            router_loss = criterion(router_logits, router_target)

            total_loss = task_loss + lambda_ortho * ortho_loss + lambda_router * router_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1
            global_step += 1
            
            if WANDB_AVAILABLE:
                log_data = {
                    "task_loss": task_loss.item(), "total_loss": total_loss.item(),
                    "router_loss": router_loss.item(), "ortho_loss": ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss,
                    "model_task_id": current_task_id, "epoch": epoch, "global_step": global_step,
                }
                wandb.log(log_data)
            
            loop.set_description(f"Data Task {config['data_task_idx']} | Model Task {current_task_id} | Epoch {epoch+1}")
            loop.set_postfix(loss=total_loss.item(), ortho=f"{ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else 0:.4f}", router=f"{router_loss.item():.4f}")

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
        if patience_counter >= patience: break
            
    print(f"🏁 Loss plateaued after {epoch} epochs.")
    
    for handle in hook_handles:
        handle.remove()
    return optimizer, global_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenformer Continual Learning')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    ### --- MODIFIED: Simplified config --- ###
    config = {
        "num_tasks": 5, "classes_per_task": 2, "batch_size": 32, "patience": 2,
        "min_delta_loss": 0.01, "lr": 1e-4, 
        "lambda_ortho": 10.0,
        "lambda_router": 1.0, 
        "attention_bonus_max": 0, "data_task_idx": 0
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if WANDB_AVAILABLE:
        wandb.init(project="tokenformer-router-cl-simplified", config=config)

    model = ContinualLearner(
        dim=128, depth=2, heads=4, mlp_dim=256,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        device=DEVICE, attention_bonus_max=config["attention_bonus_max"],
    ).to(DEVICE)
    
    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])
    criterion = nn.CrossEntropyLoss()
    ### --- MODIFIED: Single optimizer for all trainable parameters --- ###
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start_task_idx, global_step, current_task_id, results_history = 0, 0, 0, {}

    if args.resume:
        model, optimizer, start_task_idx, global_step, results_history = load_checkpoint(model, optimizer, args.resume)
        current_task_id = start_task_idx - 1 if start_task_idx > 0 else 0
    
    print(f"Total model parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    for data_task_idx in range(start_task_idx, config["num_tasks"]):
        print(f"\n--- Presenting Data from Task {data_task_idx} (Model is on Task {current_task_id}) ---")
        config["data_task_idx"] = data_task_idx

        if data_task_idx > current_task_id and data_task_idx > 0:
            current_task_id += 1
            model.grow()
            print(f"Trainable parameters after growth: {count_parameters(model, trainable_only=True):,}")
            # Re-initialize optimizer to include new parameters from growth
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            if WANDB_AVAILABLE:
                wandb.log({"growth_event": 1, "model_task_id": current_task_id, "global_step": global_step, "trainable_parameters": count_parameters(model, trainable_only=True)})
        
        ### --- MODIFIED: Simplified call to training function --- ###
        optimizer, global_step = train_until_plateau(
            model, current_task_id, train_loaders[data_task_idx], optimizer, criterion, DEVICE, 
            config["classes_per_task"], global_step, config
        )
        
        ### --- DELETED: Rehearsal buffer and sleep phase calls --- ###

        print(f"--- Finished Training on Data Task {data_task_idx} ---")
        accuracies = evaluate(model, test_loaders, DEVICE, current_task_id + 1, config["classes_per_task"])
        
        results_history[current_task_id] = accuracies
        
        state_to_save = {
            'current_task_id': current_task_id, 'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # Simplified
            'results_history': results_history,
        }
        save_checkpoint(state_to_save, filename=f"checkpoint_task_{current_task_id}_final.pth.tar")

    print_results_table(results_history, config["num_tasks"])

    if WANDB_AVAILABLE:
        wandb.summary["final_average_accuracy"] = np.mean(results_history.get(config["num_tasks"] - 1, [0]))
        wandb.summary["final_trainable_parameters"] = count_parameters(model, trainable_only=True)
        wandb.finish()