import matplotlib.pyplot as plt
import mplcursors
import numpy as np

def prepare_2D_PCA_matplotlib_fig(fig_title, pca_df, pc_data_list, show):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                            constrained_layout=True,
                            gridspec_kw={
                                'wspace': 0.1,
                                'hspace': 0.15
                            })

    fig.suptitle(f"2D PCA on {fig_title}", fontsize=30, fontweight="bold", y=0.98)

    size_column = 'photo_occurrence_count'
    max_occurrence = max(pca_df[size_column])
    marker_size_ref = 250. * max_occurrence / (40. ** 2)

    # Create a different colormap for each subplot
    colors = ['#8993f8', '#ec8272', '#44d5b2', '#bc8bf8']

    # Store cursors for each subplot
    cursors = []

    all_annotations = []

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    def hide_all_annotations():
        for ann in all_annotations:
            ann.set_visible(False)
        fig.canvas.draw_idle()

    # Add each scatter plot to the subplot grid
    for idx, (title, pc1, pc2) in enumerate(pc_data_list, 1):
        i = (idx - 1) // 2  # Calculate row: 0,0,1,1
        j = (idx - 1) % 2   # Calculate column: 0,1,0,1
        subplot_color = colors[idx-1]
        marker_sizes = pca_df[size_column] * marker_size_ref
        
        scatter = axes[i, j].scatter(
            x=pc1,
            y=pc2,
            s=marker_sizes,  # size scaling
            c=subplot_color,  # Different colors for each subplot
            linewidths=1,
            edgecolors="darkslategrey",
            alpha=0.7
        )

        axes[i, j].grid(True, linestyle='-', color='gray', linewidth=1, alpha=0.3)
        #axes[i, j].set_aspect('equal', adjustable='datalim')

        ymin, ymax, xmin, xmax = pc2.min(), pc2.max(), pc1.min(), pc1.max(),
        y_range = ymax - ymin
        x_range = xmax - xmin
        padding = y_range * 0.1  # 5% padding
        axes[i, j].set_ylim(ymin - padding, ymax + padding)
        
        # Add text labels on top of markers with offset
        for x_val, y_val, label, marker_size in zip(pc1, pc2, pca_df['label'], marker_sizes):
            # Convert marker size in points to data coordinates
            # Points are 1/72 inch, need to convert to data units
            
            # Get transform from points to data coordinates
            trans = axes[i, j].transData.inverted()
            
            # Get current figure DPI
            dpi = fig.dpi
            
            # Convert marker diameter from points to inches, then to data units
            # Marker size (s) is area in points^2, so radius is sqrt(s)/2
            marker_radius_points = np.sqrt(marker_size) / 2
            
            # Convert inches to data units (this is approximate)
            # Get the data-to-display transform
            x_display, y_display = axes[i, j].transData.transform([x_val, y_val])
            
            # Add offset in display coordinates (pixels)
            offset_pixels = marker_radius_points * 1.2  # 20% extra for spacing
            new_y_display = y_display + offset_pixels
            
            # Convert back to data coordinates
            new_x_val, new_y_val = axes[i, j].transData.inverted().transform([x_display, new_y_display])
            
            # Place text at new position
            axes[i, j].text(x_val, new_y_val, label, 
                            ha='center', va='bottom',
                            fontsize=7)
        
        # Create cursor for this subplot
        cursor = mplcursors.cursor(scatter, hover=True)
        
        # Store cursor and subplot info
        cursor.subplot_idx = (i, j)
        cursor.scatter = scatter
        cursors.append(cursor)
        
        # Create annotation handler for this specific cursor
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            label = pca_df['label'].iloc[idx]
            photo_occ = pca_df['photo_occurrence_count'].iloc[idx]
            instance_count = pca_df['instance_count'].iloc[idx]
            
            text = f"$\\bf{{Label}}$: {label}\n$\\bf{{Photo Occurrence}}$: {photo_occ:,}\n$\\bf{{Instance Count}}$: {instance_count:,}"

            sel.annotation.set_text(text)
            sel.annotation.set_backgroundcolor(subplot_color)
            sel.annotation.set_fontsize(10)
            sel.annotation.get_bbox_patch().set_boxstyle("round,pad=0.3")

            all_annotations.append(sel.annotation)

        axes[i, j].set_xlabel("")
        axes[i, j].set_ylabel("")
        axes[i, j].set_title(title, fontsize=20, pad=15)

    fig.canvas.mpl_connect('figure_leave_event', lambda e: hide_all_annotations())
    fig.canvas.mpl_connect('axes_leave_event', lambda e: hide_all_annotations())

    fig.tight_layout()

    if show is True:
        plt.show()

    return fig