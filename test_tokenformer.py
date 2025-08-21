# visualize_attention.py
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vit_pytorch import ContinualLearner # Make sure this points to your model file
from tokenformer import get_split_mnist_loaders # And your data loader file

def load_model_from_checkpoint(filepath, config, device):
    """Loads a checkpoint and reconstructs the model state."""
    print(f"Loading model from {filepath}...")
    model = ContinualLearner(
        dim=128, depth=2, heads=4, mlp_dim=256,
        num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
        device=device, attention_bonus_max = 0,
    ).to(device)
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Grow the model to the saved size before loading the state dict
    if checkpoint['current_task_id'] > 0:
        for _ in range(checkpoint['current_task_id']):
            model.grow()
            
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def visualize_routing_attention(model, image, label, task_id):
    """
    Performs a forward pass and visualizes the attention scores of the routing layer.
    """
    # The routing layer was identified as the last FFN's second Pattention layer
    # Note: The path might change if you alter the TokenformerEncoder architecture
    routing_layer = model.growing_transformer.layers[-1][1].pattn2

    # Perform a forward pass to populate the attention weights
    _ = model(image.unsqueeze(0), task_id=task_id, training=False) # Add batch dimension

    # Get the attention weights for the CLS token (at sequence position 0)
    # The shape is (1, 1, num_param_tokens) -> squeeze to (num_param_tokens)
    print(routing_layer.attn_weights.shape)
    attn_scores = routing_layer.attn_weights[0, 0, :].cpu().detach().numpy()

    # Get the boundaries of the parameters for each task
    boundaries = [0] + routing_layer.growth_indices

    # Create the plot
    plt.figure(figsize=(15, 5))
    sns.barplot(x=np.arange(len(attn_scores)), y=attn_scores, color='skyblue')

    # Add vertical lines and labels for task boundaries
    colors = ['r', 'g', 'm', 'orange']
    for i, bound in enumerate(boundaries):
        plt.axvline(x=bound - 0.5, color=colors[i % len(colors)], linestyle='--', lw=2)
        plt.text(bound + 5, max(attn_scores) * 0.9, f'Task {i} Params', color=colors[i % len(colors)], rotation=90)
        
    plt.title(f"Routing Layer Attention Scores for Input '{label}' (from Task {task_id})")
    plt.xlabel("Parameter Token Index")
    plt.ylabel("Attention Score")
    plt.tight_layout()
    plt.savefig(f"attention_visualization_task_{task_id}_label_{label}.png")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Tokenformer Routing Attention')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the model checkpoint file.')
    # parser.add_argument('--task_id', required=True, type=int, help='The task ID of the sample to visualize (e.g., 0 for digits 0,1).')
    parser.add_argument('--sample_idx', default=0, type=int, help='The index of the sample in the test set of that task.')
    args = parser.parse_args()

    # Use the same config as training for consistency
    config = { "num_tasks": 5, "classes_per_task": 2, "batch_size": 1 }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = load_model_from_checkpoint(args.checkpoint, config, DEVICE)
    
    # Get the specific data sample
    _, test_loaders = get_split_mnist_loaders(config["num_tasks"], config["classes_per_task"], config["batch_size"])

    for task_id in range(config['num_tasks']):
        image, label = test_loaders[task_id].dataset[args.sample_idx]
        image = image.to(DEVICE)

        print(f"Visualizing attention for an image of digit '{label}' from Task {task_id}...")
        visualize_routing_attention(model, image, label, task_id)