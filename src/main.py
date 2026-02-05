from dataset import *
from helpers import visualize

# ddataset_manifest_no_fd = get_detailed_dataset_manifest(drop=detailed_dataset_occurrence_counts.nlargest(5).index.tolist())#drop=dataset_occurrence_counts.nlargest(5).index.tolist())

# ddataset_manifest = get_detailed_dataset_manifest()
dataset_manifest = get_dataset_manifest()
visualize(dataset_manifest.query(gt={D_ALLUMINUM_CAN:1}))

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