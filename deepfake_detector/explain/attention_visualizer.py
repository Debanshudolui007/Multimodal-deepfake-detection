import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def generate_attention_bar_chart(weights: list[float], labels: list[str] = ["Spatial", "Frequency", "Audio"]) -> str:
    """
    Generate a bar chart of the attention weights and return it as a base64 string.
    This is useful for passing directly to the Web UI without saving to disk.
    
    Args:
        weights: List of 3 floats representing the attention weights.
        labels: The labels for the modalities.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Custom vibrant colors
    colors = ['#FF4B4B', '#00D4FF', '#00FF88']
    
    bars = ax.bar(labels, weights, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Fusion Network Modality Importance', fontsize=14, fontweight='bold', color='white')
    
    # Add value text on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
        
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    
    plt.tight_layout()
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"
