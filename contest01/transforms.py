import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

NUM_PTS = 971
CROP_SIZE = 128


class FaceHorizontalFlip(A.HorizontalFlip):
    def apply_to_keypoints(self, keypoints, **params):
        keypoints = np.array(keypoints)
        keypoints[:, 0] = (params['cols'] - 1) - keypoints[:, 0]
        lm = keypoints

        nm = np.zeros_like(lm)

        nm[:64,:]     = lm[64:128,:]     # [  0, 63]  -> [ 64, 127]:  i --> i + 64
        nm[64:128,:]  = lm[:64,:]        # [ 64, 127] -> [  0, 63]:   i --> i - 64
        nm[128:273,:] = lm[272:127:-1,:] # [128, 272] -> [128, 272]:  i --> 400 - i
        nm[273:337,:] = lm[337:401,:]    # [273, 336] -> [337, 400]:  i --> i + 64
        nm[337:401,:] = lm[273:337,:]    # [337, 400] -> [273, 336]:  i --> i - 64
        nm[401:464,:] = lm[464:527,:]    # [401, 463] -> [464, 526]:  i --> i + 64
        nm[464:527,:] = lm[401:464,:]    # [464, 526] -> [401, 463]:  i --> i - 64
        nm[527:587,:] = lm[527:587,:]    # [527, 586] -> [527, 586]:  i --> i
        nm[587:714,:] = lm[714:841,:]    # [587, 713] -> [714, 840]:  i --> i + 127
        nm[714:841,:] = lm[587:714,:]    # [714, 840] -> [587, 713]:  i --> i - 127
        nm[841:873,:] = lm[872:840:-1,:] # [841, 872] -> [841, 872]:  i --> 1713 - i
        nm[873:905,:] = lm[904:872:-1,:] # [873, 904] -> [873, 904]:  i --> 1777 - i
        nm[905:937,:] = lm[936:904:-1,:] # [905, 936] -> [905, 936]:  i --> 1841 - i
        nm[937:969,:] = lm[968:936:-1,:] # [937, 968] -> [937, 968]:  i --> 1905 - i
        nm[969:971,:] = lm[970:968:-1,:] # [969, 970] -> [969, 970]:  i --> 1939 - i

        return nm


class ToTensorV3(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "": self.apply_to_mask, "keypoints": self.apply_to_keypoints,}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = np.array(keypoints, dtype='float32')
        keypoints = torch.from_numpy(keypoints)
        return keypoints
