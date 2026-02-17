import copy
from functools import partial

import numpy


class BaseGenerator:
    """Generates data on the basis of the provided data."""
    def __init__(self, reference_data: numpy.ndarray):
    
        self.data = reference_data
        #reference_data = reference_data.values
        self.samplers_per_feature = [
            partial(numpy.random.choice, reference_data[:, feature])
            for feature in range(reference_data.shape[1])
        ]
        #print(self.samplers_per_feature)

    def __call__(self, data_to_correct: numpy.ndarray,
                 whitening_mask: numpy.ndarray,
                 whitening_strategy: str = "random",
                 k: int = 10) -> numpy.ndarray:
        """Generate data on the basis of the given `data_to_correct`.

        Args:
            whitening_mask: Boolean mask to decide which features to whiten.
            whitening_strategy: Strategy used to whiten irrelevant features. One of:
                "random": Irrelevant features are replaced with random values. Default.
                "values": Random sample from existing values.
            k: Number of samples to generate. Defaults to 10.
        """
        repetitions_per_instance = k // data_to_correct.shape[0]
        base_sample = numpy.vstack([numpy.repeat(instance, repetitions_per_instance).reshape(repetitions_per_instance, data_to_correct.shape[1])
                                    for instance in data_to_correct])
        refined_sample = copy.deepcopy(base_sample)

        if whitening_strategy == "random":
                fill = numpy.random.rand(refined_sample.shape[0], sum(whitening_mask))

        elif whitening_strategy == "values":
                fill = numpy.zeros((refined_sample.shape[0], sum(whitening_mask)))
                sampling_indices = numpy.argwhere(whitening_mask).flatten()
                #print(whitening_mask)
                #print(sampling_indices)
                
               
               

                for i, index in enumerate(sampling_indices):
                    fill[:, i] = self.samplers_per_feature[index](size=refined_sample.shape[0])

        else:
                raise ValueError(f"Unknown whitening strategy: {whitening_strategy}")

        refined_sample[:, whitening_mask] = fill

        return refined_sample
