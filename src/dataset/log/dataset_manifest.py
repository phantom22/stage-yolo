import pandas as pd
from sklearn.decomposition import PCA

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import seaborn as sns

import matplotlib.pyplot as plt

class DatasetManifest:
    def __init__(self, name, og, vec, column_ids, column_ids2labels):
        self.name = name.upper()
        self.og = og
        self.vec = vec
        self.column_ids = column_ids
        self.column_ids2labels = column_ids2labels 

        raw = pd.DataFrame(vec, columns=column_ids)
        raw_bin = raw.map(lambda x: 1 if x > 0 else 0)

        self.raw = raw.T
        self.raw_bin = raw_bin.T
        self.ccm = raw.T.dot(raw)
        self.ccm_bin = raw_bin.T.dot(raw_bin)

        self.instance_counts = raw.sum(axis=0).values
        self.photo_occurrence_counts = raw_bin.sum(axis=0).values

    def visualize_2D_PCA(self):
        pca = PCA(n_components=2)

        c_instance_count = pca.fit_transform(self.raw)
        c_photo_occurence_count = pca.fit_transform(self.raw_bin)
        c_co_intensity = pca.fit_transform(self.ccm)
        c_co_presence = pca.fit_transform(self.ccm_bin)

        pca_df = pd.DataFrame({
            'id': self.column_ids,
            'PC1': c_instance_count[:,0],
            'PC2': c_instance_count[:,1],
            'instance_count': self.instance_counts
        })

        pca_df['label'] = pca_df['id'].map(self.column_ids2labels)
        pca_df['photo_occurrence_count'] = self.photo_occurrence_counts

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
            title_text=f"2D PCA on {self.name}",
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

        print(f"[{self.name}]")
        print(f"instance_count max({max_instance_count_row['label']},{max_instance_count_row['instance_count']}), min({min_instance_count_row['label']},{min_instance_count_row['instance_count']})")
        print(f"photo_occurrence max({max_photo_occurrence_row['label']},{max_photo_occurrence_row['photo_occurrence_count']}), min({min_photo_occurrence_row['label']},{min_photo_occurrence_row['photo_occurrence_count']})")

    def visualize_CCM(self, **kw):
        ccm_renamed = self.ccm.rename(
            index=self.column_ids2labels,
            columns=self.column_ids2labels
        )

        ccm_bin_renamed = self.ccm_bin.rename(
            index=self.column_ids2labels,
            columns=self.column_ids2labels
        )
        
        show_values = kw.get("show_values", False)
        fmt = kw.get("fmt", "1.f")

        plt.figure(figsize=(12, 10))
        sns.heatmap(ccm_renamed, 
                    annot=show_values,  # Don't show values in cells
                    fmt=fmt,
                    cmap='RdBu_r',
                    center=0,    # Center colormap at 0 if values are centered
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        plt.title("Co-occurrence Matrix Heatmap")

        plt.figure(figsize=(12, 10))
        sns.heatmap(ccm_bin_renamed, 
                    annot=show_values,
                    fmt=fmt,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        plt.title("Binary Co-occurrence Matrix Heatmap")

        #plt.tight_layout()
        plt.show()


# plt.subplots(1,2,figsize=(12, 10),
#                          constrained_layout=True,
#                          gridspec_kw={
#                             'wspace':0.1,
#                             'hspace':0.15
#                          })