# Coipee
A demo implementation of the [Caipi](https://github.com/stefanoteso/caipi) explanatory interactive learning algorithm.
Coipee implements a model which one can query to retrieve uncertain instances on which it lacks confidence, alongside
an explanation of the model prediction.
The user can then correct such explanation, then feed it back to Coipee to trigger an additional training guided by
the explanation.

This implementation leverages feature masks as explanation, i.e., masks which can enable or disable input features. 


## Quickstart
Install through `pip` and `venv`:
```shell
mkvirtualenv -p python3.12 coipee

pip install coipee
```

## Usage
Coipee revolves around a `Coipee object`:
```python
barman = Coipee(
    model=base_model,  # the model to explain, e.g. a neural network
    fit_model=fit_model,  # the function to train the model, invoked after a correction
    pool=data_train,   # pool of data to measure the model's uncertainty, also used for query
    pool_labels=labels_train  # labels of the pool
)
```
A typical use involves querying the model for a number of uncertain instances
```python
number_of_instances = 10
artifact = barman.query(10)
print(artifact.explanation)
```
and retrieve a feature mask: features important to the model are marked as `True`, while others as `False`.
We can also threshold importance at different levels: the higher the threshold, the higher the required importance
to mark a feature as important:
```python
artifact = barman.query(10, threshold=0.01)
print(artifact.explanation)
```

Once we have our explanation, we can correct it by marking some important features as not important, and vice versa:
```python
import copy

corrected_artifact = copy.deepcopy(artifact)

corrected_artifact.explanation[:] = False
corrected_artifact.explanation[[0, 1, 2]] = True
```
Here, we have simply said to the model that actually, only the features `0, 1, 2` are actually important.
We can also directly retrieve differences between artifacts through the `diff` method:
```python
print(f"Difference: {artifact.diff(corrected_artifact)}")
```

Now that we have corrected the explanation, we can feed it back to the model:
```python
barman.stack_correction(corrected_artifact)  # adds the correction to correction stack of the model
barman.correct_model()  # triggers a training phase
```
