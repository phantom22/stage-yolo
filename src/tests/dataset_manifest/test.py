from setup import *


print(detailed_dataset_df.isnull().sum())
print(dataset_df.isnull().sum())
#old_dm = get_old_dataset_manifest()

#print(old_dm.raw.isnull().sum())
# old_dm.prepare_2D_PCA_fig(lib="plotly")