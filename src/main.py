from dataset import *

# dataset_manifest = get_dataset_manifest()#drop=dataset_occurrence_counts.nlargest(5).index.tolist())
# dataset_manifest.prepare_CCM_figs()
# dataset_manifest.prepare_2D_PCA_fig()

# print("Instance counts:")
# print(dataset_instance_counts.rename(index=class2txt))
# print(f"Mean instance counts: {dataset_instance_counts.drop('fd').mean()}")
# print("\nOccurrence counts:")
# print(dataset_occurrence_counts.rename(index=class2txt))
# print(f"Mean occurrence counts: {dataset_occurrence_counts.drop('fd').mean()}")

dataset_manifest_no_fd = get_dataset_manifest(drop=dataset_occurrence_counts.nlargest(5).index.tolist())#drop=dataset_occurrence_counts.nlargest(5).index.tolist())
dataset_manifest_no_fd.prepare_CCM_figs()

# d_manifest_no_fd_na = get_dataset_manifest(drop=["fd","na"])#drop=dataset_occurrence_counts.nlargest(5).index.tolist())
# d_manifest_no_fd_na.prepare_CCM_figs()
# d_manifest_no_fd_na.prepare_2D_PCA_fig()
# dd_manifest = get_detailed_dataset_manifest()
# dd_manifest.prepare_CCM_figs()

# get_detailed_dataset_manifest = get_detailed_dataset_manifest()
# get_detailed_dataset_manifest.prepare_CCM_figs(lib="plotly")

# dataset_manifest = get_dataset_manifest(drop=dataset_instance_counts.nlargest(5).index.tolist())
# print(dataset_manifest.dropped)
# dataset_manifest.prepare_CCM_figs(lib="plotly")