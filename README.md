# NSGA2-and-VEGSO-Multi-Objective-Optimization

Uma grande variedade de problemas em engenharia, indústria e muitos outros campos envolvem a otimização simultânea de vários objetivos. Em muitos casos, os objetivos são definidos em unidades incomparáveis e apresentam algum grau de conflito entre eles (ou seja, um objetivo não pode ser melhorado sem a deterioração de pelo menos outro objetivo).

O método mais comumente adotado na otimização multiobjetivo para comparar soluções é o denominado relação de dominância de Pareto que, ao invés de uma única solução ótima, leva a um conjunto de alternativas com diferentes trade-offs entre os objetivos. 
Essas soluções são chamadas de soluções ótimas de Pareto ou soluções não dominadas (non-dominated).

### Non Dominated Sorting Genetic Algorithm 2 (NSGA2)

o NSGA-II trabalha com o mesmo fluxo de um algoritmo genético padrão. 
A diferença é que, ao final de cada iteração, uma nova população para a próxima iteração é escolhida usando o método de classificação não dominado (Non Dominated Sorting).

### Vector Evaluated Glowworm Swarm Optimization (VEGSO)

O VEGSO apresenta o mesmo princípio do algoritmo VEGA (vector Evaluated Genetic Algorithm), a ideia principal é que uma parte do enxame concentra-se na função 1, e a outra parte na função 2.
Para o Algoritmo GSO essa diferenciação esta localizada na atualização do brilho (luciferin) de cada indivíduo da população.


##

En-version

# NSGA2-and-VEGSO-Multi-Objective-Optimization

A wide variety of problems in engineering, industry, and many other fields involve simultaneous optimization of multiple goals. In many cases, goals are defined in unparalleled units and present some degree of conflict between them (that is, one goal cannot be improved without the deterioration of at least one other goal).

The most commonly adopted method in multi-objective optimization to compare solutions is called in relation to Pareto dominance, which, when resulting from a single optimal solution, leads to a set of alternatives with different trade-offs between the objectives.
These solutions are called Pareto optimal solutions or Non-Dominated solutions.

### Unmastered Classification Genetic Algorithm 2 (NSGA2)

NSGA-II works with the same flow as a standard genetic algorithm.
The difference is that, at the end of each iteration, a new population for the next iteration is chosen using the non-dominated classification method (non-dominated classification).

### Vector-evaluated optimization Glowworm Swarm Optimization (VEGSO)

VEGSO presents the same principle as the VEGA (vector Evaluated Genetic Algorithm) algorithm, the main idea is that one part of the swarm concentrates on function 1, and the other part on function 2.
For the GSO Algorithm, this differentiation is located in the brightness update (luciferin) of each individual in the population.
