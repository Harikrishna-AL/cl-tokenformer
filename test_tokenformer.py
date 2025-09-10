import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tokenformer import ContinualLearner, get_split_mnist_loaders # Updated imports

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

def visualize_routing_attention(model, image, label, task_id):
    """
    Performs a forward pass and visualizes the attention scores from one of the FFN layers.
    """
    # Let's visualize the second Pattention layer of the final feed-forward block,
    # as this is likely to act as a "router" for the final representation.
    routing_layer = model.growing_transformer.layers[-1][1].pattn2

    # Perform a forward pass to populate the attention weights.
    # The `return_features` argument is no longer needed or available.
    with torch.no_grad():
        outputs = model(image.unsqueeze(0), task_id=task_id, training=False) # Add batch dimension

    # For debugging: print the model's prediction
    prediction = torch.argmax(F.softmax(outputs, dim=-1))
    print(f"Input Label: {label}, Model Prediction: {prediction.item()}")

    # Get the attention weights. Shape is (Batch, Sequence, Params).
    # Since Batch=1 and Sequence=1, we take the first element of each.
    # The shape is (1, 1, num_param_tokens) -> squeeze to (num_param_tokens)
    attn_scores = routing_layer.attn_weights[0, 0, :].cpu().detach().numpy()

    # Get the boundaries of the parameters for each task
    boundaries = [0] + routing_layer.growth_indices + [len(attn_scores)]

    # Create the plot
    plt.figure(figsize=(15, 5))
    sns.barplot(x=np.arange(len(attn_scores)), y=attn_scores, color='skyblue', saturation=0.8)

    # Add vertical lines and labels for task boundaries
    colors = ['r', 'g', 'm', 'orange', 'purple', 'brown']
    for i in range(len(boundaries) - 1):
        start_bound = boundaries[i]
        end_bound = boundaries[i+1]
        
        # Highlight the bar region for each task
        plt.axvspan(start_bound - 0.5, end_bound - 0.5, facecolor=colors[i % len(colors)], alpha=0.1)
        
        # Add a text label for the task
        text_pos = start_bound + (end_bound - start_bound) / 2
        plt.text(text_pos, max(attn_scores) * 0.9, f'Task {i}\nParams', color=colors[i % len(colors)], 
                 ha='center', va='top', fontsize=12, weight='bold')

    plt.title(f"FFN Attention Scores for Input '{label}' (from Task {task_id})", fontsize=16)
    plt.xlabel("Parameter Token Index", fontsize=12)
    plt.ylabel("Attention Score (Similarity)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"attention_visualization_task_{task_id}_label_{label}.png")
    print(f"Saved visualization to attention_visualization_task_{task_id}_label_{label}.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Tokenformer FFN Attention')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the model checkpoint file.')
    parser.add_argument('--sample_idx', default=0, type=int, help='The index of the sample in the test set of that task.')
    args = parser.parse_args()

    # Use the same config as training for consistency
    config = { "num_tasks": 5, "classes_per_task": 2, "batch_size": 1 }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model from the specified checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config, DEVICE)
    
    # Get the test dataloaders
    _, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])

    # Loop through each task, take a sample, and visualize its attention
    for task_id in range(config['num_tasks']):
        # .dataset accesses the underlying Subset, .dataset again accesses the full MNIST set
        # This is a bit complex, a simpler way is to just iterate the loader once.
        try:
            image, label = next(iter(test_loaders[task_id]))
            image = image.squeeze(0) # Remove batch dim added by loader
            label = label.item()
        except StopIteration:
            print(f"Could not get a sample from Task {task_id} loader. Skipping.")
            continue
        
        image = image.to(DEVICE)

        print(f"\n--- Visualizing attention for an image of digit '{label}' from Task {task_id} ---")
        visualize_routing_attention(model, image, label, task_id)