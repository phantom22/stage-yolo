from dataset import *

raw_pca("detailed-content-log.json", vec_detailed_dataset, variables, var2txt)
raw_pca("content-log.json", vec_dataset, classes, class2txt)