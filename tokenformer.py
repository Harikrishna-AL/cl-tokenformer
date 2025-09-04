# tokenformer_inference.py
import torch
import torch.nn as nn
import torch.optim as optim
from vit_pytorch import ContinualLearner, PattentionLayer 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from torch.distributions.multivariate_normal import MultivariateNormal

# --- DATA LOADING ---
class ContrastiveWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset; self.transform = transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return (self.transform(img), self.transform(img)), label

def get_split_mnist_loaders(num_tasks, classes_per_task, batch_size, image_size):
    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    raw_train_dataset = MNIST(root='./data', train=True, download=True)
    train_dataset = ContrastiveWrapper(raw_train_dataset, contrastive_transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=test_transform)
    train_loaders, test_loaders = [], []
    for task_id in range(num_tasks):
        start, end = task_id * classes_per_task, (task_id + 1) * classes_per_task
        task_classes = list(range(start, end))
        train_indices = [i for i, lbl in enumerate(raw_train_dataset.targets) if lbl in task_classes]
        test_indices = [i for i, lbl in enumerate(test_dataset.targets) if lbl in task_classes]
        train_loaders.append(DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))
        test_loaders.append(DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
    return train_loaders, test_loaders

# --- LOSS FUNCTIONS ---
class AngularPenaltySMLoss(nn.Module):
    def __init__(self, s=20.0, m=0.0):
        super().__init__()
        self.s = s
        self.m = m
    def forward(self, features, local_targets, classifier_head):
        weights = F.normalize(classifier_head.weight, p=2, dim=1)
        features = F.normalize(features, p=2, dim=1)
        wf = F.linear(features, weights)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[local_targets]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(local_targets)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class MahalanobisLoss(nn.Module):
    def __init__(self, sigma_inv_list):
        super(MahalanobisLoss, self).__init__()
        self.sigma_inv_list = sigma_inv_list
    def _get_pairwise_dist_matrix(self, x, cov_inv):
        gram_matrix = x @ cov_inv @ x.T
        diag = torch.diag(gram_matrix)
        dist_sq = diag.unsqueeze(1) - 2 * gram_matrix + diag.unsqueeze(0)
        return torch.sqrt(F.relu(dist_sq) + 1e-8)
    def forward(self, x_old, x_new, labels):
        unique_labels = torch.unique(labels)
        loss, n_pairs = 0.0, 0
        for label_idx, label in enumerate(unique_labels):
            class_mask = (labels == label)
            if class_mask.sum() < 2 or label_idx >= len(self.sigma_inv_list): continue
            x_old_c, x_new_c = x_old[class_mask], x_new[class_mask]
            cov_inv = self.sigma_inv_list[label_idx]
            dist_matrix_old = self._get_pairwise_dist_matrix(x_old_c, cov_inv)
            dist_matrix_new = self._get_pairwise_dist_matrix(x_new_c, cov_inv)
            abs_diff = torch.abs(dist_matrix_old - dist_matrix_new)
            loss += torch.sum(torch.triu(abs_diff, diagonal=1))
            n = x_old_c.shape[0]
            n_pairs += n * (n - 1) / 2
        return loss / n_pairs if n_pairs > 0 else 0.0

def compute_angle_weighted_patch_distillation_loss(p_n, p_o, cls_n):
    if p_n is None or p_o is None: return 0.0
    p_n_normalized = F.normalize(p_n, p=2, dim=-1)
    p_o_normalized = F.normalize(p_o.detach(), p=2, dim=-1)
    alpha_cos = F.cosine_similarity(cls_n.unsqueeze(1), p_n, dim=-1).clamp(min=-1.0, max=1.0)
    alpha_angle = 1 - (torch.acos(alpha_cos) / torch.pi)
    distances = torch.norm(p_n_normalized - p_o_normalized, p=2, dim=-1)
    weighted_distances = (1 - alpha_angle.detach()) * distances
    return weighted_distances.mean()

def info_nce_loss(z1, z2, temperature=0.1, device='cpu'):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(len(z1), device=device)
    return F.cross_entropy(sim_matrix, labels)

# --- TRAINING PHASES & HELPERS ---
def train_cssl_phase(model, model_old, data_loader, device, config, task_id):
    print(f"\n🔬 Starting Continual SSL Phase for Task {task_id}...")
    model.train(); optimizer_cssl = torch.optim.Adam(model.ssl_parameters(), lr=config["lr_cssl"])
    for epoch in range(config["cssl_epochs"]):
        loop = tqdm(data_loader, leave=False, desc=f"CSSL Epoch {epoch+1}/{config['cssl_epochs']}")
        for (view1, view2), _ in loop:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer_cssl.zero_grad()
            features1, features2 = model.get_features(view1), model.get_features(view2)
            z1, z2 = model.contrastive_projector(features1), model.contrastive_projector(features2)
            loss_simclr = info_nce_loss(z1, z2, device=device)
            total_loss = loss_simclr
            if model_old is not None:
                p1 = model.contrastive_predictor(z1)
                with torch.no_grad():
                    features_old1 = model_old.get_features(view1)
                    z_old1 = model_old.contrastive_projector(features_old1)
                loss_cassle = info_nce_loss(p1, z_old1.detach(), device=device)
                total_loss += loss_cassle
            total_loss.backward(); optimizer_cssl.step()
            loop.set_postfix(loss=total_loss.item())
    print("🔬 CSSL Phase Complete.")

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone().fill_diagonal_(0.0); mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / mask.sum() if mask.sum() > 0 else 0
    iden = torch.eye(cov.size(0), device=cov.device)
    return cov + (10 * diag_mean * iden) + (10 * off_diag_mean * (1 - iden))
def precompute_covariances(model, data_loader, device):
    # ... (this function is correct, no changes needed) ...
    print("Pre-computing covariances..."); model.eval(); all_features, all_labels = [], []
    with torch.no_grad():
        for (view1, _), labels in data_loader:
            features = model.extract_vector(view1.to(device))
            all_features.append(features); all_labels.append(labels)
    all_features, all_labels = torch.cat(all_features), torch.cat(all_labels)
    covs = []
    for label in torch.unique(all_labels):
        class_features = all_features[all_labels == label]
        if len(class_features) > 1:
            cov = torch.cov(class_features.T.double())
            covs.append(torch.linalg.pinv(shrink_cov(cov)).float().detach())
    return covs

# --- FUNCTION WITH CORRECTION ---
def update_statistics(model, data_loader, device, known_classes):
    """Calculates and returns the mean and covariance for the current task's data."""
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        # The test_loader provides clean, non-augmented data, so we can use it directly.
        # No need to rebuild a temporary loader.
        for data, labels in data_loader:
            features = model.extract_vector(data.to(device))
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    means, covs = {}, {}
    # The labels from the dataloader are the global class IDs (e.g., 2, 3 for Task 1)
    # So we can use them directly as keys.
    for label in torch.unique(all_labels):
        class_id = label.item()
        class_features = all_features[all_labels == label]
        means[class_id] = torch.mean(class_features, dim=0)
        covs[class_id] = torch.cov(class_features.T)
    return means, covs
def mean_shift_compensation(model, model_old, data_loader, stored_means, known_classes, device):
    print("🔧 Calibrating means (MSC)..."); model.eval(); model_old.eval(); all_shifts = []
    with torch.no_grad():
        for (view1, _), _ in data_loader:
            features_old = model_old.extract_vector(view1.to(device))
            features_current = model.extract_vector(view1.to(device))
            all_shifts.append((features_current - features_old).cpu())
    avg_shift = torch.mean(torch.cat(all_shifts, dim=0), dim=0)
    for i in range(known_classes): stored_means[i] += avg_shift
    print("MSC complete."); return stored_means
def classifier_alignment(model, stored_means, stored_covariances, num_tasks_seen, task_sizes, device, config):
    print("🔧 Aligning classifier..."); model.train()
    classifier_params = [p for i in range(num_tasks_seen) for p in model.mlp_heads[i].parameters()]
    optimizer = optim.SGD(classifier_params, lr=config["lr_align"], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["align_epochs"])
    total_classes = sum(task_sizes)
    for epoch in range(config["align_epochs"]):
        sampled_data, sampled_label = [], []
        for c_id in range(total_classes):
            cls_mean, cls_cov = stored_means[c_id], stored_covariances[c_id]
            jitter = torch.eye(cls_cov.shape[0]) * 1e-4
            dist = MultivariateNormal(cls_mean.float(), (cls_cov + jitter).float())
            sampled_data.append(dist.sample(sample_shape=(config["align_batch_size"],)))
            sampled_label.extend([c_id] * config["align_batch_size"])
        inputs, targets = torch.cat(sampled_data, dim=0).to(device), torch.tensor(sampled_label).long().to(device)
        sf_indexes = torch.randperm(inputs.size(0)); inputs, targets = inputs[sf_indexes], targets[sf_indexes]
        for _iter in range(total_classes):
            inp, tgt = inputs[_iter*config["align_batch_size"]:(_iter+1)*config["align_batch_size"]], targets[_iter*config["align_batch_size"]:(_iter+1)*config["align_batch_size"]]
            if len(inp) == 0: continue
            optimizer.zero_grad()
            logits = model(inp, fc_only=True)[:, :total_classes]
            per_task_norm = []
            prev_t_size = 0
            for _ti in range(num_tasks_seen):
                cur_t_size = prev_t_size + task_sizes[_ti]
                temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                per_task_norm.append(temp_norm); prev_t_size = cur_t_size
            norms = torch.cat(per_task_norm, dim=-1).mean(dim=-1, keepdim=True)
            decoupled_logits = torch.div(logits, norms) / config["logit_norm"]
            loss = F.cross_entropy(decoupled_logits, tgt)
            loss.backward(); optimizer.step()
        scheduler.step()
    print("Classifier alignment complete.")
def evaluate(model, test_loaders, device, num_tasks_seen, classes_per_task):
    model.eval(); accuracies = []
    total_classes = num_tasks_seen * classes_per_task
    with torch.no_grad():
        for task_id in range(num_tasks_seen):
            correct, total = 0, 0
            for data, target in test_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                output = model(data)[:, :total_classes]
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
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


if __name__ == '__main__':
    config = {
        "num_tasks": 5, "classes_per_task": 2, "batch_size": 128,
        "init_epoch": 10, "epochs": 10, "init_lr": 0.01, "lrate": 0.01,
        "lr_cssl": 3e-4, "cssl_epochs": 5,
        "lr_align": 0.01, "align_epochs": 10, "align_batch_size": 128,
        "num_agnostic_tokens": 64, "num_specific_tokens_per_task": 32,
        "lambda_patch_distill": 10.0, "lambda_maha": 1.0, "logit_norm": 0.1,
        "image_size": 32, "patch_size": 4, "feature_dim": 256
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ContinualLearner(
        image_size=config["image_size"], patch_size=config["patch_size"],
        dim=config["feature_dim"], depth=2, heads=4, mlp_dim=512,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        num_agnostic_tokens=config["num_agnostic_tokens"], device=DEVICE
    ).to(DEVICE)
    
    train_loaders, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"], config["image_size"])
    
    results_history, stored_means, stored_covariances, task_sizes = {}, {}, {}, []
    model_old, known_classes = None, 0

    for task_id in range(config["num_tasks"]):
        print(f"\n{'='*20} Task {task_id} {'='*20}")
        task_sizes.append(config["classes_per_task"])
        
        train_cssl_phase(model, model_old, train_loaders[task_id], DEVICE, config, task_id)
        if task_id > 0: model.grow(config["num_specific_tokens_per_task"])
        
        old_class_inv_covs = []
        if model_old is not None:
            old_class_inv_covs = precompute_covariances(model_old, train_loaders[task_id], DEVICE)

        model.train()
        lr = config["init_lr"] if task_id == 0 else config["lrate"]
        run_epochs = config["init_epoch"] if task_id == 0 else config["epochs"]
        optimizer = optim.SGD(model.supervised_parameters(task_id), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
        
        loss_cos = AngularPenaltySMLoss(s=20.0, m=0.0)
        loss_maha = MahalanobisLoss(old_class_inv_covs) if model_old is not None else None
        
        hooks = apply_masks_and_hooks(model)
        
        for epoch in range(run_epochs):
            loop = tqdm(train_loaders[task_id], leave=True, desc=f"Supervised Task {task_id} | Epoch {epoch+1}")
            for (view1, _), target in loop:
                data, target = view1.to(DEVICE), target.to(DEVICE)
                local_target = target - known_classes
                optimizer.zero_grad()
                
                features, patch_tokens = model.get_features(data, return_patch_tokens=True)

                # --- CORRECTED CALL to AngularPenaltySMLoss ---
                # The loss function computes logits internally now
                loss = loss_cos(features, local_target, model.mlp_heads[task_id])
                
                if model_old is not None and loss_maha is not None:
                    with torch.no_grad(): old_features, old_patch_tokens = model_old.get_features(data, return_patch_tokens=True)
                    loss += config["lambda_maha"] * loss_maha(old_features, features, local_target)
                    loss += config["lambda_patch_distill"] * compute_angle_weighted_patch_distillation_loss(patch_tokens, old_patch_tokens, features)

                loss.backward(); optimizer.step()
                loop.set_postfix(loss=loss.item())
            scheduler.step()
        
        for handle in hooks: handle.remove()
        
        current_means, current_covs = update_statistics(model, test_loaders[task_id], DEVICE, known_classes)
        stored_means.update(current_means); stored_covariances.update(current_covs)
        
        if model_old is not None:
            stored_means = mean_shift_compensation(model, model_old, train_loaders[task_id], stored_means, known_classes, DEVICE)
        
        classifier_alignment(model, stored_means, stored_covariances, task_id + 1, task_sizes, DEVICE, config)

        accuracies = evaluate(model, test_loaders, DEVICE, task_id + 1, config["classes_per_task"])
        results_history[task_id] = accuracies
        
        model_old = deepcopy(model).to(DEVICE).eval()
        known_classes += task_sizes[-1]