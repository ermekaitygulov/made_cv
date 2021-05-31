from typing import Dict

import segmentation_models_pytorch as smp

from utils import add_to_catalog

SEG_NN_CATALOG: Dict[str, callable] = {}


add_to_catalog('baseline', SEG_NN_CATALOG)(smp.Unet)
