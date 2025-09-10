# visualize.py
import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tokenformer import ContinualLearner, PattentionLayer # Make sure this import points to your model file

def load_model_from_checkpoint(filepath, config, device):
    """Loads a checkpoint and reconstructs the model state."""
    print(f"Loading model from {filepath}...")
    # Updated constructor to match the latest ContinualLearner __init__
    model = ContinualLearner(
        image_size=28,
        dim=128, 
        depth=2, 
        heads=4, 
        mlp_dim=256,
        num_tasks=config["num_tasks"], 
        classes_per_task=config["classes_per_task"],
        device=device, 
        attention_bonus_max = 0,
    ).to(device)
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Grow the model to the saved size before loading the state dict
    if checkpoint['current_task_id'] > 0:
        print(f"Growing model to accommodate Task {checkpoint['current_task_id']} parameters...")
        for _ in range(checkpoint['current_task_id']):
            model.grow()
            
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def _extract_and_label_tokens(model):
    """A helper function to extract tokens and their corresponding task labels."""
    try:
        # We'll visualize the tokens from a representative layer.
        target_pattention_layer = model.growing_transformer.layers[0][0].to_k
    except Exception as e:
        print(f"Could not find the target Pattention layer. Error: {e}")
        return None, None
        
    all_tokens, task_labels = [], []

    # Get the task-agnostic tokens
    # agnostic_tokens = target_pattention_layer.key_param_tokens.detach().cpu().numpy()
    # if agnostic_tokens.shape[0] > 0:
    #     all_tokens.append(agnostic_tokens)
    #     task_labels.extend([f"Agnostic ({agnostic_tokens.shape[0]})"] * agnostic_tokens.shape[0])

    # # Get the task-specific tokens and label them by task
    specific_tokens = target_pattention_layer.key_param_tokens.detach().cpu().numpy()
    boundaries = target_pattention_layer.growth_indices
    boundaries = [128]
    num_tasks_trained = len(boundaries)
    print(boundaries)
    
    
    for task_id in range(num_tasks_trained):
        # start_idx = boundaries[task_id]
        end_idx = boundaries[task_id]
        start_idx = 0
        if task_id > 0:
            start_idx = boundaries[task_id-1]
        print(start_idx, end_idx)
        num_task_tokens = end_idx - start_idx
        task_labels.extend([f"Task {task_id} ({num_task_tokens})"] * num_task_tokens)

    if specific_tokens.shape[0] > 0:
        all_tokens.append(specific_tokens)
    
    if not all_tokens:
        return None, None


    print(np.concatenate(all_tokens, axis=0)[128:256,:], len(task_labels))
    print(np.concatenate(all_tokens, axis=0)[:128,:])

        
    return np.concatenate(all_tokens, axis=0)[:128,:], task_labels


def visualize_token_distribution(model, title="Key Parameter Token Distribution (t-SNE)"):
    """Visualizes the distribution of key parameter tokens via a t-SNE scatter plot."""
    print("\n🔬 Visualizing key parameter token distribution (t-SNE)...")
    
    all_tokens_np, token_task_labels = _extract_and_label_tokens(model)
    if all_tokens_np is None:
        print("No tokens to visualize.")
        return

    print(f"Running t-SNE on {all_tokens_np.shape[0]} tokens of dimension {all_tokens_np.shape[1]}...")
    perplexity_value = min(30.0, all_tokens_np.shape[0] - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, 
                init='pca', learning_rate='auto')
    tokens_2d = tsne.fit_transform(all_tokens_np)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    plot = sns.scatterplot(x=tokens_2d[:, 0], y=tokens_2d[:, 1], hue=token_task_labels,
                           palette="viridis", s=50, alpha=0.8)
    
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plot.legend(title="Token Group")
    
    output_filename = "token_distribution_tsne.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE visualization saved to {output_filename}")


def visualize_token_value_distribution(model, title="Key Parameter Token Value Distribution (KDE)"):
    """Visualizes the distribution of the values within the tokens using a KDE plot."""
    print("\n🔬 Visualizing key parameter token value distribution (KDE)...")
    
    all_tokens_np, token_task_labels = _extract_and_label_tokens(model)
    if all_tokens_np is None:
        print("No tokens to visualize.")
        return

    # Flatten the tokens into a single long vector of individual parameter values
    all_values_flat = all_tokens_np.flatten()
    
    # Create a parallel array of labels, repeating the task label for each value in a token
    feature_dim = all_tokens_np.shape[1]
    labels_for_values = np.repeat(token_task_labels, feature_dim)
    
    plt.figure(figsize=(12, 8))
    plot = sns.kdeplot(x=all_values_flat, hue=labels_for_values, palette="magma", fill=True, alpha=0.6)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Parameter Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True)
    
    output_filename = "token_value_distribution_kde.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ KDE visualization saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Tokenformer Parameters')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the model checkpoint file.')
    args = parser.parse_args()

    # This config should contain the necessary architectural details of the saved model.
    # It's important that these match the model that was trained.
    config = {
        "num_tasks": 5, "classes_per_task": 2,
        "image_size": 32, "patch_size": 4, "feature_dim": 256,
        "num_agnostic_tokens": 64, "num_specific_tokens_per_task": 32,
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model from the specified checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config, DEVICE)
    
    # Generate both visualizations
    visualize_token_distribution(model)
    visualize_token_value_distribution(model)