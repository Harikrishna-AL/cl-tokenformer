# visualize_pdf.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import argparse
import os

# You must be able to import IncrementalPCA from your ipca.py file
from ipca import IncrementalPCA

def visualize_marginal_pdfs(ipca_path, class_to_visualize, n_components_to_show=4, output_path=None, device='cpu'):
    """
    Loads a saved iPCA model for a specific class and visualizes the
    marginal Probability Density Function (PDF) along its top principal components.
    """
    if not os.path.exists(ipca_path):
        print(f"Error: iPCA model file not found at {ipca_path}")
        print("Please run Phase 1 of the main script first.")
        return

    print(f"Loading iPCA models from {ipca_path}...")
    ipca_states = torch.load(ipca_path, map_location=device)
    
    if class_to_visualize not in ipca_states:
        print(f"Error: Class {class_to_visualize} not found in the saved iPCA models.")
        print(f"Available classes are: {sorted(list(ipca_states.keys()))}")
        return

    # 1. Load the specific iPCA model for the target class
    state_dict = ipca_states[class_to_visualize]
    ipca_model = IncrementalPCA(n_components=state_dict['n_components'], device=device)
    ipca_model.load_state_dict(state_dict)
    print(f"Successfully loaded iPCA model for class {class_to_visualize}.")

    # 2. Extract the distribution parameters
    mean_vec = ipca_model.mean.cpu().numpy()
    components = ipca_model.components.cpu().numpy()
    variances = ipca_model.explained_variance.cpu().numpy()
    std_devs = np.sqrt(variances)
    
    n_components_to_show = min(n_components_to_show, ipca_model.n_components)

    # 3. Create a plot with subplots for each component
    fig, axes = plt.subplots(n_components_to_show, 1, figsize=(8, 2.5 * n_components_to_show), sharex=False)
    if n_components_to_show == 1:
        axes = [axes] # Make it iterable if there's only one subplot

    fig.suptitle(f'Marginal Probability Density Functions for Class {class_to_visualize}', fontsize=16, y=0.95)

    for i in range(n_components_to_show):
        ax = axes[i]
        std_dev = std_devs[i]
        
        # Create a range of values around the mean for plotting
        # The distribution along a principal component is centered at 0 with std_dev
        x = np.linspace(-4 * std_dev, 4 * std_dev, 1000)
        
        # Calculate the PDF of the normal distribution
        pdf = norm.pdf(x, loc=0, scale=std_dev)
        
        # Plotting
        ax.plot(x, pdf, lw=2)
        ax.fill_between(x, pdf, alpha=0.2)
        ax.set_title(f'Principal Component {i+1} (StDev: {std_dev:.3f})')
        ax.set_ylabel('Probability Density')
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Value along Principal Component Axis')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 4. Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"PDF visualization saved successfully to {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the marginal PDF of a learned feature distribution.')
    parser.add_argument('--ipca_path', type=str, default='checkpoints/ipca_models.pth',
                        help='Path to the saved Incremental PCA models file.')
    parser.add_argument('--class_id', required=True, type=int,
                        help='The class ID you want to visualize (e.g., 0, 1, 2...).')
    parser.add_argument('--components', type=int, default=4,
                        help='Number of top principal components to show.')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Optional path to save the output plot. If not provided, the plot is displayed directly.')
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    output_filename = args.output_path
    if output_filename is None:
        # Create a default filename if none is provided
        output_filename = f"pdf_visualization_class_{args.class_id}.png"

    visualize_marginal_pdfs(
        ipca_path=args.ipca_path,
        class_to_visualize=args.class_id,
        n_components_to_show=args.components,
        output_path=output_filename,
        device=DEVICE
    )