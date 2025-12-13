import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def fig(fig_title, pca_df, pc_data_list, show):
    
    # Create a 2x2 grid of subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Instance counts',
            f'Photo Occurrence',
            f'Co-Intensity',
            f'Co-Presence'
        ),
        horizontal_spacing=0.025,
        vertical_spacing=0.075
    )


    size_column = 'photo_occurrence_count'
    max_occurrence = max(pca_df[size_column])
    marker_size_ref = 2.*max_occurrence/(40.**2)

    # Add each scatter plot to the subplot grid
    for idx, (_, pc1, pc2) in enumerate(pc_data_list, 1):
        row = (idx - 1) // 2 + 1  # Calculate row:    1,1,2,2
        col = (idx - 1) % 2 + 1   # Calculate column: 1,2,1,2
        
        trace = go.Scatter(
            x=pc1,
            y=pc2,
            mode='markers+text',
            text=pca_df['label'],
            textposition='top center',
            customdata=pca_df[['photo_occurrence_count', 'instance_count']],
            hovertemplate=(
                "<b>Label:</b> %{text}<br>"
                "<b>Photo Occurrence:</b> %{customdata[0]:,}<br>"
                "<b>Instance Count:</b> %{customdata[1]:,}<br>"
                "<extra></extra>"
            ),
            marker=dict(
                size=pca_df[size_column],
                sizemode='area',
                sizeref=marker_size_ref,
                sizemin=4,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            showlegend=False
        )
        
        fig.add_trace(trace, row=row, col=col)

    # Update the layout for the whole page
    fig.update_layout(
        autosize=True,
        title_text=f"2D PCA on {fig_title}",
        title_font=dict(size=30,weight="bold"),
        title_x=0.5,
        title_y=0.98,
        title_xanchor="center",
        title_yanchor="top",
        showlegend=False,
        margin=dict(l=0, r=0, t=80, b=80)
    )

    # Remove axis tooltips
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    # Update subplot title style
    fig.update_annotations(
        font=dict(size=28), 
        yshift=7
    )

    if show is True:
        fig.show(config={'responsive': True})

    return fig