import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, ROOT_DIR)

try:
    from dataset import *
except ImportError as e:
    print(f"Error importing dataset: {e}")

from tests.dataset_manifest.dataset_manifest_new import DatasetManifestNew
from tests.dataset_manifest.dataset_manifest_old import DatasetManifestOld

def get_new_detailed_dataset_manifest(**kw):
    return DatasetManifestNew("annotation_manifest", 
                           detailed_dataset_df, detailed_dataset_bin_df, 
                           detailed_dataset_ccm, detailed_dataset_bin_ccm,
                           detailed_dataset_instance_counts, detailed_dataset_occurrence_counts,
                           variables, var2txt, **kw)

def get_new_dataset_manifest(**kw):
    return DatasetManifestNew("stripped_annotation_manifest",
                           dataset_df, dataset_bin_df, 
                           dataset_ccm, dataset_bin_ccm,
                           dataset_instance_counts, dataset_occurrence_counts,
                           classes, class2txt, **kw)

def get_old_detailed_dataset_manifest(**kw):
    return DatasetManifestOld("annotation_manifest", 
                           detailed_dataset_df, variables, var2txt, **kw)

def get_old_dataset_manifest(**kw):
    return DatasetManifestOld("stripped_annotation_manifest",
                           dataset_df, classes, class2txt, **kw)