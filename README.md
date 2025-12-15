## About this implementation
This programme implements evolutionary strategy known as ES(μ + λ).  
Genotype consists of 2 arrays of n real numbers. First is about deciding what to invest in, second is used for mutatation. In order to extract fenotype (where and how much to invest) conversion is needed: extract k highest numbers from genotype and normalize them (sum = 1), while change other numbers to 0.
This allows for freedom in crossover and mutation operators.

Used components:
- initialization - random with population size as a parameter
- evaluation - calculated as in the project description; in case of not fulfilling constraints, penalty is applied
- stop criteria - deadline time
- reproduction - random
- crossover - average in extended version for both arrays in genotype; Allows for good exploration
    For every gene position, it generates a random value a ∈ [0,1] and then computes the child’s gene as a weighted average of the two parents
    child_gene = a * parent1_gene + (1 − a) * parent2_gene
- mutation - typical mutation for this algorithm;  
    σi ← σi exp(τ′a + τbi), where σi is single gene in individual's genotype for mutation, τ = 1/√(2n), τ′ = 1/√(2√(n)), a = N(0, 1), bi = N(0, 1) for i in 1..n  
    M ← x + σi * N(0, 1), where x is individual's genotype for investing
- succession - elite with elite size equal to the size of population;
    best individuals from both partens' and children's population combined  
    (possible also ES(μ, λ) algorithm where next generation is created only from children's population)
    


## Things to consider:
- try with different operators (reproduction, crossover, mutation, succession, stop criteria)
- find optimal parameters
- go back to basic evolutionary algorithm and try with different operators and parameters there
- different behaviour in hard tasks (for example if n=1000 then focus more on exploration)
- try to preprocess sth and use that information
- different population initialization (maybe some greedy heuristics)
- round solutions close to best to achive best possible globally
- ...
