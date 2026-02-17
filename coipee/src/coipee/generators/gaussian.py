import copy
from typing import Tuple

import numpy
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture


class GaussianGenerator:
    """Generates data on an induced multivariate Gaussian kernel, correcting generated data on the basis of the
     given conditions."""
    def __init__(self, reference_data: numpy.ndarray, **kwargs):
        self.data = reference_data
        self.model = GaussianMixture(**kwargs)

        self.model.fit(reference_data)

        self.neighbor_imputer = KNNImputer(n_neighbors=5, weights="distance", missing_values=numpy.nan)
        self.neighbor_imputer.fit(self.data)
        self.whitening_values = {
            "mean": self.data.mean(axis=0),
            "zero": numpy.zeros(self.data.shape[1],)
        }

    def __call__(self, whitening_mask: numpy.ndarray,
                 whitening_strategy: str = "mean",
                 k: int = 10) -> Tuple[numpy.ndarray, numpy.array]:
        """Sample `k` instances from this Gaussian generator.

        Args:
            whitening_mask: Boolean mask to decide which features to whiten.
            whitening_strategy: Strategy used to whiten irrelevant features. One of:
                "mean": Irrelevant features are replaced with the reference data mean. Default value.
                "zero": Irrelevant features are replaced with 0.
                "neighbor": Irrelevant features are replaced with the average of neighbors.
            k: Number of samples to generate. Defaults to 10.
        """
        base_sample, base_labels = self.model.sample(k)  #todo: check if also labels can be sampled
        refined_sample = copy.deepcopy(base_sample)

        if  whitening_strategy == "mean" or  whitening_strategy == "zero":
            #case "mean" | "zero":
                refined_sample[:, whitening_mask] = self.whitening_values[whitening_strategy][whitening_mask]
        elif  whitening_strategy ==  "neighbor":
                refined_sample[:, whitening_mask] = numpy.nan

                refined_sample= self.neighbor_imputer.transform(refined_sample)
        else :
                raise ValueError(f"Unknown whitening strategy: {whitening_strategy}")

        return refined_sample, base_labels
