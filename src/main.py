from dataset import *

detailed_dataset_manifest = get_detailed_dataset_manifest(drop=detailed_dataset_instance_counts.nlargest(5).index.tolist())
print(detailed_dataset_manifest.dropped)
detailed_dataset_manifest.prepare_CCM_figs(lib="plotly")


dataset_manifest = get_dataset_manifest(drop=dataset_instance_counts.nlargest(5).index.tolist())
print(dataset_manifest.dropped)
dataset_manifest.prepare_CCM_figs(lib="plotly")