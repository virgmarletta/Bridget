import numpy
from shap import KernelExplainer


class FeatureImportanceExplainer:
    """A KernelSHAP-based feature importance explainer."""
    def __init__(self, model, reference_data: numpy.ndarray):
        """
        Args:
            model: The model to explain.
            reference_data: The reference data.
        """
        self.reference_data = reference_data
        self.max_samples = self.reference_data.shape[0]

        # todo: remove the 100
       
        self.explainer = KernelExplainer(model.predict, self.reference_data[:100])

    def explain(self, data: numpy.ndarray, threshold: float = 0.05) -> numpy.ndarray:
        """Compute the feature importance of the given `data`, returning a boolean mask: features
        above `threshold` are deemed important, features below are not."""
        base_values = abs(self.explainer.shap_values(data, nsamples=100))
        explanation = base_values > threshold

        return explanation
