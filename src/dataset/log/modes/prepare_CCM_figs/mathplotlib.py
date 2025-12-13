import seaborn as sns
import matplotlib.pyplot as plt

def fig(ccm, ccm_bin, show_values, fmt, show):
    fig1 = plt.figure(figsize=(12, 10))
    sns.heatmap(ccm, 
                annot=show_values,  # Don't show values in cells
                fmt=fmt,
                cmap='RdBu_r',
                center=0,    # Center colormap at 0 if values are centered
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Co-occurrence Matrix Heatmap")
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(12, 10))
    sns.heatmap(ccm_bin, 
                annot=show_values,
                fmt=fmt,
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Binary Co-occurrence Matrix Heatmap")
    fig2.tight_layout()

    if show is True:
        plt.show()

    return fig1, fig2