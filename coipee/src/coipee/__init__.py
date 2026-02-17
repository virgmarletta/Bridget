from __future__ import annotations


from typing import Callable, Tuple, List, Optional

import numpy

from coipee.src.coipee.artifacts import FeatureMaskArtifact
from coipee.src.coipee.detectors import Selector
from coipee.src.coipee.explanations import FeatureImportanceExplainer
from coipee.src.coipee.generators.base import BaseGenerator
from coipee.src.coipee.generators.gaussian import GaussianGenerator


class Coipee:
    """The Coipee system for feature importance artifacts:

    - `Coipee.query` to retrieve a number of uncertain instances, and their explanation
    - `Coipee.stack_correction` to add the correction to the instance's stack of corrections
    - `Coipee.correct_model` to apply corrections

    """
    def __init__(self, model, fit_model: Callable, pool: numpy.ndarray, pool_labels: numpy.ndarray,
                 names: Optional[List[str]] = None):
        self._correction_stack: List[FeatureMaskArtifact] = list()

        self.model = model
        self.fit_model = fit_model
        self.names = names

        self._pool: numpy.ndarray = pool
        self._pool_labels: numpy.array = pool_labels

        self._explainer: FeatureImportanceExplainer = FeatureImportanceExplainer(model=model, reference_data=pool)
        #print('Explainer :  self._explainer', self._explainer)
        self._corrected_data_generator: BaseGenerator = BaseGenerator(pool)
        
        self._confidence_sampler = Selector(model=model)
        
     
    def _sample(self, k: int = 10) -> Tuple[numpy.ndarray, numpy.ndarray]:
        #print('Sono qui')
        uncertainty_indices = self._confidence_sampler(self._pool, k)
        #print(f"Pool size: {self._pool.shape[0]}, Requested k: {k}")
        print( uncertainty_indices)
        return self._pool[uncertainty_indices], self._pool_labels[uncertainty_indices]

    def query(self, number_of_instances: int = 10, threshold: float = 0.005) -> FeatureMaskArtifact:
        """Query the model for uncertain instances, retrieving an explanation which can
        be amended and used to update the model.

        Args:
            number_of_instances: Number of returned instances.
            threshold: Threshold to consider a feature as important.

        Returns:
           An artifact which can be corrected and returned to Coipee to
            update the model.
        """
        pooled_instances, pooled_labels = self._sample(k=number_of_instances)
        pooled_explanations = self._explainer.explain(pooled_instances, threshold=threshold)
        explanation = pooled_explanations.mean(axis=0) > 0.5

        return FeatureMaskArtifact(explanation=explanation, explained_data=pooled_instances,
                                   explained_labels=pooled_labels, names=self.names)
    



    def query_1(self, pooled_instances,pooled_labels, threshold: float = 0.005) -> FeatureMaskArtifact:
        """Query the model for uncertain instances, retrieving an explanation which can
        be amended and used to update the model.
        Args:
        number_of_instances: Number of returned instances.
        threshold: Threshold to consider a feature as important.
        Returns:
       An artifact which can be corrected and returned to Coipee to
        update the model.
        """
        #pooled_instances, pooled_labels = self._sample(k=number_of_instances)
        pooled_explanations = self._explainer.explain(pooled_instances, threshold=threshold)
        explanation = pooled_explanations.mean(axis=0) > 0.5
        return FeatureMaskArtifact(explanation=explanation, explained_data=pooled_instances,
                               explained_labels=pooled_labels, names=self.names)

    def generate_counterexamples(self, data_to_correct: numpy.ndarray, labels: numpy.array,
                                 correction_mask: numpy.array) -> Tuple[numpy.ndarray, numpy.ndarray]:
        counterexamples = self._corrected_data_generator(data_to_correct=data_to_correct,
                                                         whitening_mask=correction_mask,
                                                         whitening_strategy="values",
                                                         k=100 * data_to_correct.shape[0])
        labels = numpy.concatenate([numpy.array([label] * 100) for label in labels])

        return counterexamples, labels

    def correct_model(self) -> Coipee:
        """Correct the model on the given data.

        Returns:
            This Coipi instance.
        """
        for artifact in self._correction_stack:
            counterexamples, labels = self.generate_counterexamples(data_to_correct=artifact.explained_data,
                                                                    labels=artifact.explained_labels,
                                                                    correction_mask=artifact.explanation)

            self.model = self.fit_model(self.model, counterexamples, labels)

        # reset stack
        self._correction_stack = list()

        return self

    def stack_correction(self, correction: FeatureMaskArtifact) -> Coipee:
        """Add the provided explanation correction to this instance's stack. At any point, corrections can
        be used to update the model by invoking self.correct. The stack allows for repeated corrections,
        it is *not* a sorted set."""
        self._correction_stack.append(correction)

        return self
