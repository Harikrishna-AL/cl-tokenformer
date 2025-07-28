from vit_pytorch import ViT, TokenformerViT, PattentionLayer
import torch

config = {
        "num_tasks": 5,
        "classes_per_task": 2,
        "batch_size": 64,
        "epochs_per_task": 3,
        "lr": 1e-4,
        "perplexity_growth_threshold": 3.0, # ### NEW ### Grow if perplexity is 3x the EMA
        "ema_alpha": 0.1, # ### NEW ### Smoothing factor for EMA
        "data_task_idx": 0 # For logging purposes
    }
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = TokenformerViT(
    image_size=28, patch_size=7, num_tasks=config["num_tasks"], classes_per_task=config["classes_per_task"],
    channels=1, dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1, emb_dropout=0.1, device=DEVICE
).to(DEVICE)


print(model.transformer.layers[1][1].net[1])
rand = torch.rand((28,28)).to(DEVICE)
out = model(rand, 0)

attn_weights = model.transformer.layers[1][1].net[1].attn_weights.numpy()
plt.imshow(attn_weights, cmap='viridis', interpolation='nearest')  # You can change the colormap (cmap)

# Add a colorbar
plt.colorbar()

# Add labels and title (optional)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("My Heatmap")

# Save the image
plt.savefig("heatmap" + str(data_task_idx) + ".png")