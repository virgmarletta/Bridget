import copy
import pprint
from typing import List, Optional, Set

import numpy


class FeatureMaskArtifact:
    """An interaction artifact based on feature whitening."""
    def __init__(self, explanation: numpy.array, explained_data: numpy.ndarray, explained_labels: numpy.ndarray,
                 names: Optional[List[str]] = None):
        """

        Args:
            explanation: The explanation, a boolean mask on the features
            explained_data: Data that was explained by this artifact
            explained_labels: Labels of `explained_data`
            names: Optional feature names
        """
        self.explanation = explanation
        self.explained_data = explained_data
        self.explained_labels = explained_labels
        self.names = names

    def diff(self, other) -> Set:
        """Difference with the other feature mask artifact."""
        if not isinstance(other, FeatureMaskArtifact):
            raise ValueError(f"Expected a FeatureMaskArtifact, {type(other)} found.")

        indices = numpy.argwhere(self.explanation != other.explanation).squeeze()

        if self.names is None:
            return set(indices)
        else:
            return set((self.names[i], i) for i in indices)

    def __hash__(self):
        return hash(self.explained_data.data)

    def __copy__(self):
        return FeatureMaskArtifact(explanation=self.explanation, explained_data=self.explained_data,
                                   explained_labels=self.explained_labels, names=self.names)

    def __deepcopy__(self, memdict = {}):
        return FeatureMaskArtifact(explanation=copy.deepcopy(self.explanation),
                                   explained_data=copy.deepcopy(self.explained_data),
                                   explained_labels=copy.deepcopy(self.explained_labels),
                                   names=copy.deepcopy(self.names))

    def __str__(self):
        base = f"Number of explained instances: {self.explained_data.shape[0]}"

        if self.names is None:
            base += "\n" + pprint.pformat(self.explanation.astype(str))
        else:
            base += "\nExplanation:"
            base += "\n" + pprint.pformat(dict(zip(self.names, self.explanation)))

        return base
