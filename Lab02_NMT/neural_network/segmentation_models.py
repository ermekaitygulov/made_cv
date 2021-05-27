from typing import Type, Dict

import segmentation_models_pytorch as smp
from torch.nn import Module

from utils import add_to_catalog

SEG_NN_CATALOG: Dict[str, Type[Module]] = {}


add_to_catalog('baseline', SEG_NN_CATALOG)(smp.Unet)
