import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def raw_pca(dataset_label, dataset_as_rows, column_ids, column_ids2labels):
    dataset_label_u = dataset_label.upper()
    pca = PCA(n_components=2)
    
    df_counts = pd.DataFrame(dataset_as_rows, columns=column_ids)
    df_binary = df_counts.map(lambda x: 1 if x > 0 else 0)

    # transpose so features (classes) are rows
    c_instance_count = pca.fit_transform(df_counts.T)
    c_photo_occurence_count = pca.fit_transform(df_binary.T)
    # co-occurence matrix of the instance counts
    c_co_intensity = pca.fit_transform(df_counts.T.dot(df_counts))
    # co-occurence matrix of the photo occurrence counts
    c_co_presence = pca.fit_transform(df_binary.T.dot(df_binary))

    pca_df = pd.DataFrame({
        'id': column_ids,
        'PC1': c_instance_count[:,0],
        'PC2': c_instance_count[:,1],
        'instance_count': df_counts.sum(axis=0).values
    })

    pca_df['label'] = pca_df['id'].map(column_ids2labels)
    pca_df['photo_occurrence_count'] = df_binary.sum(axis=0).values

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

    # Define the PCA data for each subplot
    pc_data_list = [
        (c_instance_count[:, 0], c_instance_count[:, 1]),
        (c_photo_occurence_count[:, 0], c_photo_occurence_count[:, 1]),
        (c_co_intensity[:, 0], c_co_intensity[:, 1]),
        (c_co_presence[:, 0], c_co_presence[:, 1])
    ]

    size_column = 'photo_occurrence_count'
    max_occurrence = max(pca_df[size_column])
    marker_size_ref = 2.*max_occurrence/(40.**2)

    # Add each scatter plot to the subplot grid
    for idx, (pc1, pc2) in enumerate(pc_data_list, 1):
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
                "<b>Instance Count:</b> %{customdata[0]:,}<br>"
                "<b>Photo Occurrence:</b> %{customdata[1]:,}<br>"
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
        title_text=f"{dataset_label_u}",
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

    # Show the combined figure
    fig.show(config={'responsive': True})

    max_photo_occurrence_row = pca_df.loc[pca_df['photo_occurrence_count'].idxmax()]
    min_photo_occurrence_row = pca_df.loc[pca_df['photo_occurrence_count'].idxmin()]

    max_instance_count_row = pca_df.loc[pca_df['instance_count'].idxmax()]
    min_instance_count_row = pca_df.loc[pca_df['instance_count'].idxmin()]

    print(f"[{dataset_label_u}]")
    print(f"instance_count max({max_instance_count_row['label']},{max_instance_count_row['instance_count']}), min({min_instance_count_row['label']},{min_instance_count_row['instance_count']})")
    print(f"photo_occurrence max({max_photo_occurrence_row['label']},{max_photo_occurrence_row['photo_occurrence_count']}), min({min_photo_occurrence_row['label']},{min_photo_occurrence_row['photo_occurrence_count']})")