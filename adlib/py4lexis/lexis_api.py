from py4lexis.session import LexisSession
from py4lexis.ddi.datasets import Datasets

p4l = LexisSession("lexis_config.toml")
datasets = Datasets(p4l)
print("all datasets:")
for i, dataset_info in enumerate(datasets.get_all_datasets()[0]):
    print(i, dataset_info)

print("\ndataset status:")
print(datasets.get_dataset_status())