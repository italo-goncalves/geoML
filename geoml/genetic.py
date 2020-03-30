# geoML - machine learning models for geospatial data
# Copyright (C) 2019  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as _np
# from scipy.spatial import distance_matrix as _distance_matrix


def real_mutation_radial(parent, temperature):
    """
    Treats the chromossome as coordinates in Euclidean space, shifting it
    a small distance in a random direction.
    """
    min_val = parent - temperature
    max_val = parent + temperature
    
    child = parent + _np.random.uniform(min_val, max_val, len(parent))
    child[child < 0] = 0
    child[child > 0] = 0
    return child


def real_mutation_parallel(parent, temperature):
    """
    Treats the chromosome as individual values. The temperature parameter
    controls the probability of change for each value. If a value is changed,
    it can take any value in its range.
    """
    # new values
    child = _np.random.uniform(0, 1, len(parent))
    
    # controls which values will be changed
    change = _np.random.choice([0, 1], size=len(parent), replace=True,
                               p=(1 - temperature, temperature))
    return parent*(1-change) + child*change


def real_mutation_step(parent, temperature):
    """
    Treats the chromossome as coordinates in Euclidean space, shifting it
    0.1% in a random direction.
    """
    min_val = parent - 0.001
    max_val = parent + 0.001

    child = parent + _np.random.uniform(min_val, max_val, len(parent))
    child[child < 0] = 0
    child[child > 0] = 0
    return child


MUTATIONS_REAL = (real_mutation_radial,
                  real_mutation_parallel,
                  real_mutation_step,
                  )


def crossover_1p(parent1, parent2):
    """One-point crossover"""
    xp = _np.random.choice(len(parent1) - 1, 1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[_np.arange(0, xp)] = parent2[_np.arange(0, xp)]
    child2[_np.arange(0, xp)] = parent1[_np.arange(0, xp)]
    return child1, child2


def crossover_2p(parent1, parent2):
    """Two-point crossover"""
    if len(parent1) < 3:
        return crossover_1p(parent1, parent2)
    
    xp1 = _np.random.choice(len(parent1) - 2, 1)
    xp2 = _np.random.choice(_np.arange(xp1 + 1, len(parent1) - 1), 1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[_np.arange(xp1, xp2 + 1)] = parent2[_np.arange(xp1, xp2 + 1)]
    child2[_np.arange(xp1, xp2 + 1)] = parent1[_np.arange(xp1, xp2 + 1)]
    return child1, child2


def real_crossover_average(parent1, parent2):
    """Returns two weighted averages of the parents, with random weights."""
    frac = _np.random.uniform(0, 1, 2)
    
    child1 = frac[0]*parent1 + (1-frac[0])*parent2
    child2 = frac[1]*parent1 + (1-frac[1])*parent2
    return child1, child2


CROSSOVERS_REAL = (crossover_1p,
                   crossover_2p,
                   real_crossover_average)


def integer_mutation_parallel(parent, temperature, n_vals):
    """
    Treats the chromosome as individual values. The temperature parameter
    controls the probability of change for each value. If a value is changed,
    it can take any value in its range.
    """

    # controls which values will be changed
    change = _np.random.choice([0, 1], size=len(parent), replace=True,
                               p=(1 - temperature, temperature))
    change = _np.where(change == 1)[0]

    if change.shape[0] == 0:
        return parent

    vals = _np.arange(n_vals)
    for pos in change:
        parent[pos] = _np.random.choice(vals, size=1)[0]
    return parent


INTEGER_MUTATIONS = (integer_mutation_parallel)


# def _find_min_count(d):
#     """
#     For a given distance matrix, finds the minimum distance that gives at
#     least a count of 1 neighbor for all points.
#     """
#     if d.sum() == 0:
#         return 0
#
#     d2 = _np.tril(d, -1)
#     d2 = d2[d2 > 0]
#
#     for i in range(1000):
#         x = (d2.max() - d2.min())*i/1000 + d2.min()
#         count = (d <= x).sum(0) - 1  # discounting the diagonal
#         if count.min() == 1:
#             break
#     return x
#
#
# def _share_fitness(popmatrix, fitvec):
#     """
#     Shares fitness based on the number of neighbors within the radius
#     that gives at least 1 neighbor for each individual.
#
#     Returns a vector with positive values representing relative probability
#     for selection.
#     """
#     # d = _distance_matrix(popmatrix, popmatrix)
#
#     # cosine distance
#     n_var = popmatrix.shape[1]
#     popmatrix = popmatrix*2 - 1
#     d = 1 - _np.matmul(popmatrix, _np.transpose(popmatrix)) / n_var
#
#     dmin = _find_min_count(d) + 1e-6
#
#     # sh contains the distances above the threshold
#     sh = _np.ones_like(d)
#     sh = sh - d/dmin
#     sh[sh < 0] = 0
#
#     # weights
#     w = sh.sum(0)
#
#     amp = fitvec.max() - fitvec.min()
#     if amp == 0:
#         amp = 10
#     fitvec = fitvec - fitvec.min() + 0.1*amp
#     p = fitvec / w + 1e-6
#     p = p / _np.sum(p)
#     return p


class GeneticOptimizer:
    def __init__(self, generator, fitness, mutations, crossovers):
        self.generator = generator
        self.fitness = fitness
        self.popsize = 50
        self.mutations = mutations
        self.crossovers = crossovers
        self.mut_prob = 0.35
        self.seed = None
        self.population = []
        self.best_solution = None
        self.best_fitness = None
        self.evolution_log = []
        self.verbose = True
        self.temperature = 0.5

    def populate(self, start=None):
        """Resets population"""
        if self.seed is not None:
            _np.random.seed(self.seed)

        self.population = start if start is not None else []
        while len(self.population) < self.popsize:
            self.population.append(self.generator())

        fit_list = []
        for i, elem in enumerate(self.population):
            if self.verbose:
                print("\rInitializing population: " + str(i) + "       ",
                      end="")
            fit_list.append(self.fitness(elem))
        if self.verbose:
            print("\n")

        fitvec = _np.array(fit_list)
        self.evolution_log = [fitvec]
        self.best_fitness = _np.max(fitvec)
        self.best_solution = self.population[int(_np.argmax(fitvec))]

    def iterate(self, replace_worst=False):
        """One iteration of training"""
        # selection
        fitvec = self.evolution_log[-1].copy()
        # p = _share_fitness(popmatrix, fitvec)
        p = fitvec - fitvec.min()
        p = p + 0.1 * p.max() + 1e-6
        p = p / p.sum()
        sel = _np.random.choice(self.popsize, 2, replace=False, p=p)

        # crossover/mutation
        crossfun = _np.random.choice(self.crossovers)
        child1, child2 = crossfun(self.population[sel[0]],
                                  self.population[sel[1]])

        mutfun = _np.random.choice(self.mutations)
        if _np.random.uniform() < self.mut_prob:
            child1 = mutfun(child1, self.temperature)
        if _np.random.uniform() < self.mut_prob:
            child2 = mutfun(child2, self.temperature)

        # updating population
        if replace_worst:
            # replace smallest fitness - more aggressive
            tmp_fitness = self.fitness(child1)
            pos = _np.where(fitvec == fitvec.min())[0][0]
            if tmp_fitness > fitvec[pos]:
                fitvec[pos] = tmp_fitness
                self.population[pos] = child1
            tmp_fitness = self.fitness(child2)
            pos = _np.where(fitvec == fitvec.min())[0][0]
            if tmp_fitness > fitvec[pos]:
                fitvec[pos] = tmp_fitness
                self.population[pos] = child2
        else:
            # replacing parents if better - preserves diversity
            tmp_fitness = self.fitness(child1)
            if (fitvec[sel[0]] < fitvec[sel[1]]) & \
                    (fitvec[sel[0]] < tmp_fitness):
                fitvec[sel[0]] = tmp_fitness
                self.population[sel[0]] = child1
            elif (fitvec[sel[1]] < fitvec[sel[0]]) & \
                    (fitvec[sel[1]] < tmp_fitness):
                fitvec[sel[1]] = tmp_fitness
                self.population[sel[1]] = child1
            tmp_fitness = self.fitness(child2)
            if (fitvec[sel[0]] < fitvec[sel[1]]) & \
                    (fitvec[sel[0]] < tmp_fitness):
                fitvec[sel[0]] = tmp_fitness
                self.population[sel[0]] = child2
            elif (fitvec[sel[1]] < fitvec[sel[0]]) & \
                    (fitvec[sel[1]] < tmp_fitness):
                fitvec[sel[1]] = tmp_fitness
                self.population[sel[1]] = child2

        # updating log
        self.evolution_log.append(fitvec)
        self.best_fitness = _np.max(fitvec)
        self.best_solution = self.population[int(_np.argmax(fitvec))]

    def optimize(self, max_iter=1000, popsize=50, mut_prob=0.35, tol=0.1,
                 patience=None, seed=None, start=None):
        self.popsize = popsize
        self.mut_prob = mut_prob
        self.seed = seed

        if patience is None:
            patience = _np.ceil(max_iter / 5)

        self.populate(start)

        stagnation = 0
        self.temperature = 1
        best_fitness = self.best_fitness
        if self.verbose:
            print("Iteration 0 | Best fitness = " + str(self.best_fitness),
                  end="")
        for i in range(max_iter):
            self.iterate(replace_worst=i / max_iter > 0.7)

            ev = self.best_fitness - best_fitness
            if ev < tol:
                stagnation += 1
                self.temperature = _np.maximum(0.05, _np.minimum(
                    self.temperature + 0.001, (1 - i / max_iter) ** 2))
            else:
                stagnation = 0
                self.temperature = 0.05
            if stagnation >= patience:
                if self.verbose:
                    print("\nTerminating training at iteration " + str(i + 1))
                break
            else:
                best_fitness = self.best_fitness
                if self.verbose:
                    print("\rIteration: " + str(i + 1) + " | Best fitness: "
                          + str(self.best_fitness) + " | Improvement <"
                          + str(tol) + " for " + str(stagnation) + "/"
                          + str(patience) + " iterations        ",
                          sep="", end="")


class GeneticOptimizerReal(GeneticOptimizer):
    def __init__(self, fitness, length, mutations=None, crossovers=None):
        def generator():
            # All real chromosomes are normalized to lie in the [0,1] interval
            return _np.random.uniform(size=[length])

        if mutations is None:
            mutations = MUTATIONS_REAL
        if crossovers is None:
            crossovers = CROSSOVERS_REAL
        super().__init__(generator, fitness,
                         mutations, crossovers)
