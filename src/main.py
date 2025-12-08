from dataset import *
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

dataset_as_vectors = []
for e in dataset:
    if e is None:
        continue
    vec = {}
    for id in ids:
        vec[id] = e.get(id, 0)  # Get count, default to 0
    dataset_as_vectors.append(vec)

df_counts = pd.DataFrame(dataset_as_vectors, columns=ids)

pca = PCA(n_components=2)
# transpose so features (classes) are rows
components = pca.fit_transform(df_counts.T)

pca_df = pd.DataFrame({
    'class': ids,
    'PC1': components[:,0],
    'PC2': components[:,1],
    'instance_count': df_counts.sum(axis=0).values
})

pca_df['class_name'] = pca_df['class'].map(id2txt)

fig1 = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    text='class_name',
    size='instance_count',
    title='PCA Visualization of Food/Item Classes (Based on Co-occurrence)',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
)

fig1.update_traces(
    textposition = 'top center',
    marker = {
        "line": {
            "width":1,
            "color":'DarkSlateGrey'
        }
    }
)
fig1.update_layout(showlegend=False)
fig1.show()


# Create binary version (truncate counts >1 to 1)
df_binary = df_counts.map(lambda x: 1 if x > 0 else 0)
components_binary = pca.fit_transform(df_binary.T)

# Update dataframe
pca_df['PC1'] = components_binary[:, 0]
pca_df['PC2'] = components_binary[:, 1]
pca_df['photo_occurrence'] = df_binary.sum(axis=0).values

fig2 = px.scatter(
    pca_df,
    x='PC1', y='PC2',
    text='class_name',
    size='photo_occurrence',
    title='PCA on Binary Co-occurrence (Presence/Absence Only)',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
)
fig2.update_traces(
    textposition = 'top center',
    marker = {
        "line": {
            "width":1,
            "color":'DarkSlateGrey'
        }
    }
)
fig2.update_layout(showlegend=False)
fig2.show()
