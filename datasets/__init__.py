# from .casc_ifw import CascIfwDataModule
from datasets.fayoum_banana import FayoumBananaDataModule
# from datasets.apple import AppleDataModule

AVAILABLE_DATASETS = {
    # "casc_ifw": CascIfwDataModule,
    "fayoum": FayoumBananaDataModule
    # "apple": AppleDataModule
}
