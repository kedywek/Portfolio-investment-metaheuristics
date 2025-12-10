## Pre-Assignment
Current pre-assignment implementation clusters the assets based on cosine distance of the distance matrix columns. After clustering the asset with highest return within a cluster is chosen and the assets within a distance threshold are excluded from the rest of the search.

Reduces the dimensionality of the search by mutually excluding entries that are similar to each other. Assets that have low cosine distance will contribute very similarly to the objective function.

## About this implementation
Genotype consists of n real numbers. In order to extract fenotype (where and how much to invest) conversion is needed: extract k highest numbers from genotype and normalize them (sum = 1), while change other numbers to 0.
This allows for freedom in crossover and mutation operators.

Used components:
- initialization - random with population size as a parameter
- evaluation - calculated as in the project description; in case of not fulfilling constraints, penalty is applied
- stop criteria - deadline time
- reproduction - tournament; Good for exploration
- crossover - average in extended version; Allows for good exploration
    For every gene position, it generates a random value a ∈ [0,1] and then computes the child’s gene as a weighted average of the two parents
    child_gene = a * parent1_gene + (1 − a) * parent2_gene
- mutation - gaussian with sigma as a parameter;
    x = x + σ · N(0, 1) where N is normal (gaussian) distribution
- succession - elite with elite size = 1;
    new_gen = best_parent + children - worst individual from this combination; Good for exploatation


## Things to consider:
- try with different operators (reproduction, crossover, mutation, succession, stop criteria)
- find optimal parameters
- different behaviour in hard tasks (for example if n=1000 then focus more on exploration)
- try to preprocess sth and use that information
- different population initialization (maybe some greedy heuristics)
- round solutions close to best to achive best possible globally
- ...
