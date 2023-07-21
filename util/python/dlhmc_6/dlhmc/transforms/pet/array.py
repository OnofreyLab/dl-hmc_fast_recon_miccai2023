from typing import Any, List, Optional, Sequence

import torch
import numpy as np
import pandas as pd

from monai.config.type_definitions import NdarrayOrTensor

from monai.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Randomizable,
    Resized,
    Transform,
)

from dlhmc.utils import Relative_motion_A_to_B_12, RotTransMatrix_6Params


__all__ = [
    "ComputeRelativeMotion",
    "CreateImageStack",
    "RandSamplePET",
]


class CreateImageStack(Transform):
    """
    Stack single frame Nd images into a (N+1)d image tensor.

    This method assumes that all input data is contained in a Pandas DataFrame, which is used 
    as the main input to the function. Image path information can be found in the DataFrame column
    specified by the `image_key` parameter.

    Args:
        image_key: string specifying the column in the input DataFrame that contains image paths.
        spatial_size: (uses functionality from monai.transforms.Resize) expected shape of spatial 
            dimensions after resize operation. If some components of the `spatial_size` are non-positive 
            values, the transform will use the corresponding components of img size. For example, 
            `spatial_size=(32, -1)` will be adapted to `(32, 64)` if the second spatial dimension size 
            of img is `64`.

    """

    def __init__(
        self, 
        image_key: str,
        spatial_size: Optional[Sequence[int]] = None, 
    ) -> None:
        self.image_key = image_key
        self.spatial_size = spatial_size


    def __call__(self, data_frame: pd.DataFrame,
    ) -> torch.Tensor:

        # Create transforms for loading and resizing
        keys = [self.image_key]

        load_transforms = Compose([
            LoadImaged(keys=keys, reader='ITKReader'),
            EnsureChannelFirstd(keys=keys),
            Resized(keys=keys, spatial_size=self.spatial_size),
        ])

        input_dict = data_frame.to_dict(orient="records")
        load_ds = Dataset(data=input_dict, transform=load_transforms)

        # Iterate through each image and append to list
        images = list()

        for batch in load_ds:
            images.append(batch[self.image_key])

        # Stack the set of images 
        stack = torch.stack(images)
        return stack







class RandSamplePET(Randomizable):
    """
    Customized data sampler for PET imaging data with ground-truth motion information.
    PET imaging data consists of a single 4d image stack with corresponding rigid motion 
    transformation parameters.

    The returned sample consists of the following:
        image_ref: the reference image from time t_ref (from the 4d image)
        image_mov: the moving image from time t_mov (from the 4d image)
        t_ref: the reference image time (from the df)
        t_mov: the moving image time (from the df)
        transformation: the six rigid transformations describing the motion between image_ref and image_mov
            (computed from the df)

    """

    def __init__(
        self, 
        num_samples: int = 1,
        meta_keys: Optional[Sequence[int]] = None,
        image_only: bool = False,
    ) -> None:
        """
        TODO: Document me!
        """
                
        self.num_samples = num_samples
        self.meta_keys = meta_keys
        self.image_only = image_only

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSamplePET":
            super().set_random_state(seed, state)
            return self


    def randomize(self, data: Optional[Any] = None) -> None:
        pass

    def __call__(self, image: torch.Tensor, meta_info: Optional[pd.DataFrame] = None
    ) -> List[torch.Tensor]:
        """
        TODO: Document me!
        """

        n = image.shape[0]

        M = np.arange(n*n).reshape(n,n)
        M = np.triu(M)
        I_valid = M[M>0]

        # TODO: set random state for the choice function
        idx = np.sort(np.random.choice(I_valid, size=self.num_samples, replace=False))
        idx_pairs = np.unravel_index(idx, shape=M.shape)

        ref_idx = idx_pairs[0]
        mov_idx = idx_pairs[1]

        results = []
        for r, m in zip(ref_idx, mov_idx):
            sample = []

            sample.append(image[r,...])
            sample.append(image[m,...])

            if meta_info is not None and self.meta_keys is not None:
                df_meta_info_ref = meta_info.iloc[r]
                df_meta_info_mov = meta_info.iloc[m]

                for key in self.meta_keys:
                    sample.append(df_meta_info_ref[key])
                    sample.append(df_meta_info_mov[key])

            results.append(sample)

        return results



class ComputeRelativeMotion(Transform):
    """
    Compute the relative motion between two sets of motion transformations.

    This method assumes that the motion is in 12-parameter (affine) format.

    Args:

    """

    def __init__(
        self, 
    ) -> None:
        return

    def __call__(self, ref: NdarrayOrTensor, mov: NdarrayOrTensor
    ) -> NdarrayOrTensor:

        #relative_motion = Relative_motion_A_to_B_12(ref, mov)
        relative_motion = RotTransMatrix_6Params(Relative_motion_A_to_B_12(ref, mov), 1)
        print(relative_motion.shape)
        return relative_motion



