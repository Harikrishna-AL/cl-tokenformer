# visualize_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vit_pytorch import ContinualLearner # Make sure this points to your EDITED model file
from tokenformer import get_split_mnist_loaders # Using the data loader from the main script

def load_model_from_checkpoint(filepath, config, device):
    """Loads a checkpoint and reconstructs the model state."""
    print(f"Loading model from {filepath}...")
    
    # ### EDITED: Removed attention_bonus_max to match new model constructor ###
    model = ContinualLearner(
        dim=128, depth=2, heads=4, mlp_dim=256,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        device=device,
    ).to(device)
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if checkpoint['current_task_id'] > 0:
        for _ in range(checkpoint['current_task_id']):
            model.grow()
            
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def visualize_routing_attention(model, image, label, task_id):
    """
    Performs a forward pass and visualizes the attention scores of a Pattention layer.
    """
    # Path to a Pattention layer to visualize.
    # e.g., the second pattn layer in the first feed-forward block of the encoder.
    routing_layer = model.growing_transformer.layers[0][1].pattn2

    # ### EDITED: Updated the forward pass call and how the return values are unpacked ###
    # The forward pass now returns 4 items. The feature vector is the 3rd one.
    outputs, _, separated_feature_vector, _ = model(image.unsqueeze(0), task_id=task_id, training=False)
    
    # Check if the attention weights were saved (requires the edit in tokenformer.py)
    if routing_layer.attn_weights is None:
        print("ERROR: attn_weights not found in the PattentionLayer. Make sure you've edited tokenformer.py to save them.")
        return None

    # Get the attention weights for the CLS token (at sequence position 0)
    attn_scores = routing_layer.attn_weights[0, 0, :].cpu().detach().numpy()

    boundaries = [0] + routing_layer.growth_indices

    # Create the plot
    plt.figure(figsize=(15, 5))
    sns.barplot(x=np.arange(len(attn_scores)), y=attn_scores, color='skyblue')

    colors = ['r', 'g', 'm', 'orange', 'purple']
    for i, bound in enumerate(boundaries):
        plt.axvline(x=bound - 0.5, color=colors[i % len(colors)], linestyle='--', lw=2)
        plt.text(bound + 5, max(attn_scores) * 0.9, f'Task {i} Params', color=colors[i % len(colors)], rotation=90)
        
    plt.title(f"P-Attention Scores for Input '{label}' (from Task {task_id})")
    plt.xlabel("Parameter Token Index")
    plt.ylabel("Raw Attention Score (pre-GELU)")
    plt.tight_layout()
    filename = f"attention_visualization_task_{task_id}_label_{label}.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close() # Close the plot to avoid displaying it in non-interactive environments
    
    return separated_feature_vector # Return the feature vector for the orthogonality check


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Tokenformer Routing Attention')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the model checkpoint file.')
    parser.add_argument('--sample_idx', default=0, type=int, help='The index of the sample in the test set of that task.')
    args = parser.parse_args()

    config = { "num_tasks": 5, "classes_per_task": 2, "batch_size": 1 }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(args.checkpoint, config, DEVICE)
    
    _, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])

    features = []
    for task_id in range(config['num_tasks']):
        # Ensure the test loader and dataset are not empty
        if len(test_loaders[task_id].dataset) > args.sample_idx:
            image, label = test_loaders[task_id].dataset[args.sample_idx]
            image = image.to(DEVICE)

            print(f"\nVisualizing attention for an image of digit '{label}' from Task {task_id}...")
            feat = visualize_routing_attention(model, image, label, task_id)
            if feat is not None:
                features.append(feat)
        else:
            print(f"Task {task_id} has fewer than {args.sample_idx + 1} samples. Skipping.")

    print("\n--- Checking Pairwise Feature Orthogonality ---")
    if len(features) > 1:
        for i in range(len(features)):
            for j in range(i, len(features)):
                # This is the same function used during training
                ortho_loss = calculate_feature_orthogonality_loss(torch.cat([features[i], features[j]], dim=0))
                print(f"Orthogonality loss between Task {i} and Task {j} features: {ortho_loss.item():.6f}")