import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_gradient_flow(grad_dict, title, save_path):
    # Process layer names and data
    layers = [k.replace('.weight', '').replace('.bias', '') for k in grad_dict.keys()]
    num_epochs = len(next(iter(grad_dict.values())))
    data = np.array([grad_dict[k] for k in grad_dict.keys()])
    
    # Compute layer-wise means and standard deviations
    means = [np.mean(grad_dict[k]) for k in grad_dict.keys()]
    stds = [np.std(grad_dict[k]) for k in grad_dict.keys()]
    
    # Create a figure with 1 row and 2 columns, setting the relative widths
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(16, len(layers) * 0.4 + 3),
        gridspec_kw={'width_ratios': [70, 30]}
    )
    
    # Plot heatmap on the left subplot (remove the colorbar/sidebar with cbar=False)
    ax0 = axes[0]
    sns.heatmap(
        data, 
        ax=ax0, 
        xticklabels=range(1, num_epochs+1), 
        yticklabels=layers, 
        cmap='viridis', 
        cbar=False
    )
    ax0.set_xlabel('Epoch')
    ax0.set_title('Gradient Flow Heatmap')
    # Ensure y-axis tick labels appear on the left side (default)
    ax0.yaxis.tick_left()
    ax0.yaxis.set_label_position("left")
    
    # Plot horizontal bar plot on the right subplot
    ax1 = axes[1]
    y_positions = np.arange(len(layers))
    ax1.barh(y_positions, means, xerr=stds, capsize=5, color='skyblue')
    ax1.set_yticks(y_positions)
    # Remove y-axis tick labels from the layer-wise plot
    ax1.set_yticklabels([])
    ax1.invert_yaxis()  # Display the first layer at the top
    ax1.set_xlabel('Gradient Norm')
    ax1.set_title('Layer-wise Gradient Flow')
    
    # Remove extra top and bottom margins:
    bar_height = 0.8  # Default bar height in matplotlib barh
    ax1.set_ylim((len(layers) - 1) + bar_height / 2, 0 - bar_height / 2)
    
    # Set the overall title for the combined plot
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def generate_gradient_flow_plots(models):
    for md in models.values():    
        model = md[0]    
        # Assume that the json contains a key 'train_grad_norms' that is a dictionary
        # mapping layer names to gradient norm lists.
        grad_norms = model.metrics._raw['grad_l2']
        
        # Split the gradients into weights and biases, excluding some layers from the process.
        weight_grads = {k: v for k, v in grad_norms.items() 
                        if 'weight' in k and 'bias' not in k and 'bn' not in k and 'norm' not in k}
        bias_grads   = {k: v for k, v in grad_norms.items() 
                        if 'bias' in k and 'bn' not in k and 'norm' not in k}
        
        # Generate combined plot for weights.
        weight_combined_title = f"Gradient Flow of Weights for {model.model_name}"
        weight_combined_save = os.path.join(model.log_dir, f"gradient_flow_weight.png")
        plot_gradient_flow(weight_grads, weight_combined_title, weight_combined_save)
        
        # Generate combined plot for biases.
        bias_combined_title = f"Gradient Flow of Bias for {model.model_name}"
        bias_combined_save = os.path.join(model.log_dir, f"gradient_flow_bias.png")
        plot_gradient_flow(bias_grads, bias_combined_title, bias_combined_save)