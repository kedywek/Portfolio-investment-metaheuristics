# Using pre-assignment to limit search space dimensionality

The pre-assignment is used to limiti the search space to speed up the convergence of the metaheuristics. This is a guide on how to use the implementation in any new metaheuristic implementation.

## 1. Import

Start by importing the `PreAssignmentMixin` from `pre_assingment_mixin.py`.

```python
from pathlib import Path
import sys
# Ensure we can import from templates/
# Assuming this file is in a subdir of the git directory
this_dir = Path(__file__).resolve().parent.parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))
from templates.pre_assignment_mixin import PreAssignmentMixin
```

## 2. Inheritance

The `PreAssignmentMixin` works by having your new metaheuristic inherit while defining as shown below.

```python
class YourMetaheuristic(PreAssignmentMixin):
    ...
```

## 3. Running

To run the pre-assignment simply run `self.apply_pre_assignment()` after reading the instance in `self.run()`.

## 4. Setting solutions

To ensure that solutions are set with the same dimensionality as the loaded instance use `self.expand_weights()` when setting the best solution. Something like the following after any processing of the solution such as normailzing in this case:

```python
def get_best_solution(self):
    ...
    # If pre-assignment reduced the universe, expand back
    expanded = self.expand_weights(normalized) if self.pre_ass else normalized
    return expanded.tolist()
```