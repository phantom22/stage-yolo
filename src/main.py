from dataset import *
from helpers import visualize

# ddataset_manifest_no_fd = get_detailed_dataset_manifest(drop=detailed_dataset_occurrence_counts.nlargest(5).index.tolist())#drop=dataset_occurrence_counts.nlargest(5).index.tolist())

# ddataset_manifest = get_detailed_dataset_manifest()
what_to_preserve = set([DD_COCA_COLA, DD_CHINOTTO, DD_THE_PESCA, DD_SPRITE, DD_MONSTER, DD_FOCO_MARACUYA])
full_set = set(variables)
ddataset_manifest = get_detailed_dataset_manifest()# drop=list(full_set - what_to_preserve))
ddataset_manifest.prepare_2D_PCA_fig()
# visualize(ddataset_manifest.query(
#     gt={
#         DD_LICORICE:1
#     },
#     lt={ 
#         DD_MILKA:0,
#         DD_KELLOGS:0,
#         DD_CRACKERS:0
#     }
# ))

# visualize(ddataset_manifest.query(
#     gt={
#         DD_PARCHMENT_PAPER:1,
#         DD_RED_TRAY:1
#     },
#     lt=None
# ))

# visualize(ddataset_manifest.query(
#     gt={
#         DD_PARCHMENT_PAPER:1,
#         DD_GRAY_TRAY:1
#     },
#     lt=None
# ))