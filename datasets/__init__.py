from .casc_ifw import CascIfwDataModule
from .fayoum_banana import FayoumBananaDataModule

AVAILABLE_DATASETS = {
    "casc_ifw": CascIfwDataModule,
    "fayoum": FayoumBananaDataModule,
}