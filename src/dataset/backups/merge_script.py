# How to merge 2 different CVAT tasks (thier main annotation tasks):
# 1.in each job, select the main annotation task
# 2.do Export annotations and choose 'Datumaro 1.0' as export format
# 3.unzip both exports and used them in this script
# 4.zip the contents of the merged folder
# 5.create a new job and choose the same project as before
# 6.create a new task and upload all images that were used in the two merged tasks
# 7.select the task
# 8.do Import annotations and choose 'Datumaro 1.0' as import format

import datumaro as dm
# Load datasets from exported zips
dataset1 = dm.Dataset.import_from('A', format='datumaro')
dataset2 = dm.Dataset.import_from('B', format='datumaro')
# Merge datasets
merged_dataset = dm.Dataset.from_extractors(dataset1, dataset2)
# Export the combined result
merged_dataset.export('merged', format='datumaro')