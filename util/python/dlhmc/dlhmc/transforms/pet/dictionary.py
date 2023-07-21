
from typing import Any, List, Optional, Sequence


from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy

from monai.transforms import MapTransform
from monai.transforms import Randomizable

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor



from dlhmc.transforms.pet.array import (
    ComputeRelativeMotion,
    CreateImageStack,
    RandSamplePET,
)


__all__ = [
    "ComputeRelativeMotiond",
    "ComputeRelativeMotionD",
    "ComputeRelativeMotionDict",
    "CreateImageStackd",
    "CreateImageStackD",
    "CreateImageStackDict",
    "RandSamplePETd",
    "RandSamplePETD",
    "RandSamplePETDict",
]


class CreateImageStackd(MapTransform):
    """Stack single frame Nd images into a (N+1)d image tensor.

    """
    
    backend = CreateImageStack.backend

    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        spatial_size: Optional[Sequence[int]] = None, 
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.image_key = image_key
        self.create_image_stack = CreateImageStack(image_key=image_key, spatial_size=spatial_size)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            d[self.image_key] = self.create_image_stack(d[key])
            
        return d
    




class RandSamplePETd(Randomizable, MapTransform):
    
    # backend = RandSamplePET.backend

    
    def __init__(
        self, 
        keys: KeysCollection,
        num_samples: int = 1,
        meta_data_key: Optional[str] = None,
        meta_keys: Optional[Sequence[str]] = None,
        ref_suffix: str = "_ref", 
        mov_suffix: str = "_mov",
        image_only: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.num_samples = num_samples
        self.meta_data_key = meta_data_key
        self.meta_keys = meta_keys
        self.ref_suffix = ref_suffix
        self.mov_suffix = mov_suffix
        self.image_only = image_only

        self.sampler = RandSamplePET(num_samples=num_samples, meta_keys=meta_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(self.sampler.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.sampler.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        for key in self.key_iterator(d):
            for i, im in enumerate(self.sampler(d[key], meta_info=d[self.meta_data_key])):
                # ret[i][key] = im
                ret[i][key+self.ref_suffix] = im[0]
                ret[i][key+self.mov_suffix] = im[1]

                for j, meta_key in enumerate(self.meta_keys):
                    ret[i][meta_key+self.ref_suffix] = im[2*j+2]
                    ret[i][meta_key+self.mov_suffix] = im[2*j+3]
    
        return ret



class ComputeRelativeMotiond(MapTransform):
    """
    Compute the relative motion between two sets of motion transformations.

    Args:
        keys: string specifying the reference and moving Vicra parameters. Assume both are 
            arrays of 12 parameters.
        output_key: string specifying the new key to create for the relative Vicra result.
            If None provided (default), the output key will be a "foo_bar" based on the
            provided keys "foo" and "bar" for the reference and moving motion parameters, 
            respectively.

    """
    
    def __init__(
        self,
        keys: KeysCollection,
        output_key: str = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.output_key = output_key
        self.compute_motion = ComputeRelativeMotion()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # TODO: Check for two keys
        ref_key = self.keys[0]
        mov_key = self.keys[1]

        out_key = ref_key+"_"+mov_key
        if self.output_key is not None:
            out_key = self.output_key

        d[out_key] = self.compute_motion(d[ref_key],d[mov_key])
        return d



CreateImageStackD = CreateImageStackDict = CreateImageStackd
RandSamplePETD = RandSamplePETDict = RandSamplePETd
ComputeRelativeMotionD = ComputeRelativeMotionDict = ComputeRelativeMotiond
