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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as _np
from scipy.spatial import distance_matrix as _distance_matrix

# All real chromosomes are normalized to lie in the [0,1] interval


def real_mutation_radial(parent, temperature):
    """
    Treats the chromossome as coordinate in Euclidean space, shifting it
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


def real_crossover_1p(parent1, parent2):
    """One-point crossover"""
    xp = _np.random.choice(len(parent1) - 1, 1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[_np.arange(0, xp)] = parent2[_np.arange(0, xp)]
    child2[_np.arange(0, xp)] = parent1[_np.arange(0, xp)]
    return child1, child2


def real_crossover_2p(parent1, parent2):
    """Two-point crossover"""
    if len(parent1) < 3:
        return real_crossover_1p(parent1, parent2)
    
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

    
def _find_min_count(d):
    """
    For a given distance matrix, finds the minimum distance that gives at
    least a count of 1 neighbor for all points.
    """
    if d.sum() == 0:
        return 0
    
    d2 = _np.tril(d, -1)
    d2 = d2[d2 > 0]
    
    for i in range(1000):
        x = (d2.max() - d2.min())*i/1000 + d2.min()
        count = (d <= x).sum(0) - 1  # discounting the diagonal
        if count.min() == 1:
            break
    return x


def _share_fitness(popmatrix, fitvec):
    """
    Shares fitness based on the number of neighbors within the radius
    that gives at least 1 neighbor for each individual.
  
    Returns a vector with positive values representing relative probability
    for selection.
    """
    # d = _distance_matrix(popmatrix, popmatrix)

    # cosine distance
    n_var = popmatrix.shape[1]
    popmatrix = popmatrix*2 - 1
    d = 1 - _np.matmul(popmatrix, _np.transpose(popmatrix)) / n_var

    dmin = _find_min_count(d) + 1e-6
    
    # sh contains the distances above the threshold
    sh = _np.ones_like(d)
    sh = sh - d/dmin
    sh[sh < 0] = 0
    
    # weights
    w = sh.sum(0)
    
    amp = fitvec.max() - fitvec.min()
    if amp == 0:
        amp = 10
    fitvec = fitvec - fitvec.min() + 0.1*amp
    p = fitvec / w + 1e-6
    p = p / _np.sum(p)
    return p

    
def training_real(fitness, minval, maxval, popsize=50,
                  mutation=(real_mutation_parallel, real_mutation_radial),
                  crossover=(real_crossover_1p, real_crossover_2p, real_crossover_average),
                  mut_prob=0.35, max_iter=1000,
                  tol=0.1, stopping=None,
                  seed=None,
                  start=None, verbose=True):
    """
    Genetic optimizer for real-valued chromosomes.

    This genetic algorithm works by selecting two individuals from the
    population, performing crossover and mutation. The new individuals replace
    the ones with the smallest fitness.
    
    Individuals are selected using fitness sharing to preserve diversity.

    Parameters
    ----------
    fitness : function
        A function that takes an array vector as the only parameter and returns
        a scalar. The function to be maximized.
    minval : array
        The lower limits for the chromosomes.
    maxval : array
        The upper limits for the chromosomes.
    popsize : int
        Number of individuals in the population.
    mutation : tuple or list
        The allowed mutation functions.
    crossover : tuple or list
        The allowed crossover functions.
    mut_prob : double
        The probability for mutation.
    max_iter : int
        The maximum number of iterations.
    tol : double
        Minimum expected improvement of the fitness function. An improvement
        less than tol does not count as an improvement for the algorithm's
        counter.
    stopping : int
        Maximum number of iterations without improvement. The algorithm will
        stop if this value is reached.
    seed : int
        Optional seed number.
    start : array
        Optional starting point for the algorithm. Will be included in the
        initial population.
    verbose : bool
        Whether to print information on console.

    Returns
    -------
    out : dict
        A dictionary with the following items:
            best_sol : array
                The best solution found by the algorithm.
            best_fitness : double
                The optimized fitness value.
            last_pop : array
                The state of the population when the algorithm stops.
            evolution : array
                The fitness values for the whole population for each iteration.

    """
    # checking
    if len(minval) != len(maxval):
        raise ValueError("maxval and minval have different lengths")
    if start is not None:
        if len(start) != len(maxval):
            raise ValueError("wrong length for starting point")
    if stopping is None:
        stopping = _np.ceil(max_iter / 5)
    if seed is not None:
        _np.random.seed(seed)
        
    # initialization
    val_length = len(maxval)
    popmatrix = _np.random.uniform(0, 1, (popsize, val_length))
    if start is not None:
        start = (start - minval)/(maxval - minval + 1e-9)
        popmatrix[0, :] = start
    
    fitvec = _np.zeros(popsize)
    if verbose:
        print("Initializing population", end="")
    for i in range(popsize):
        if verbose:
            print(".", end="")
        fitvec[i] = fitness(popmatrix[i, :]*(maxval - minval) + minval)
    if verbose:
        print("\n")
    fitmatrix = _np.zeros((popsize, max_iter + 1))
    fitmatrix[:, 0] = fitvec
 
    best_fitness = fitvec.max()
    best_sol = popmatrix[_np.where(fitvec == best_fitness)[0], :]
    
    # main loop
    stagnation = 0
    temperature = 1
    if verbose: 
        print("Iteration 0 | Best fitness = " + str(best_fitness), end="")
    for i in range(max_iter):
        # selection
        # p = _share_fitness(popmatrix, fitvec)
        p = fitvec - fitvec.min()
        p = p + 0.1*p.max() + 1e-6
        p = p/p.sum()
        sel = _np.random.choice(popsize, 2, replace=False, p=p)
        
        # crossover/mutation
        crossfun = _np.random.choice(crossover)
        child1, child2 = crossfun(popmatrix[sel[0], :], popmatrix[sel[1], :])
        
        mutfun = _np.random.choice(mutation)
        if _np.random.uniform() < mut_prob:
            child1 = mutfun(child1, temperature)
        if _np.random.uniform() < mut_prob:
            child2 = mutfun(child2, temperature)
        
        # update population
        if i/max_iter > 0.7:
            # replace smallest fitness - more aggressive
            tmp_fitness = fitness(child1*(maxval - minval) + minval)
            pos = _np.where(fitvec == fitvec.min())[0][0]
            if tmp_fitness > fitvec[pos]:
                fitvec[pos] = tmp_fitness
                popmatrix[pos, :] = child1
            tmp_fitness = fitness(child2*(maxval - minval) + minval)
            pos = _np.where(fitvec == fitvec.min())[0][0]
            if tmp_fitness > fitvec[pos]:
                fitvec[pos] = tmp_fitness
                popmatrix[pos, :] = child2
        else:
            # replacing parents if better - preserves diversity
            tmp_fitness = fitness(child1 * (maxval - minval) + minval)
            if (fitvec[sel[0]] < fitvec[sel[1]]) & \
                    (fitvec[sel[0]] < tmp_fitness):
                fitvec[sel[0]] = tmp_fitness
                popmatrix[sel[0], :] = child1
            elif (fitvec[sel[1]] < fitvec[sel[0]]) & \
                    (fitvec[sel[1]] < tmp_fitness):
                fitvec[sel[1]] = tmp_fitness
                popmatrix[sel[1], :] = child1
            tmp_fitness = fitness(child2 * (maxval - minval) + minval)
            if (fitvec[sel[0]] < fitvec[sel[1]]) & \
                    (fitvec[sel[0]] < tmp_fitness):
                fitvec[sel[0]] = tmp_fitness
                popmatrix[sel[0], :] = child2
            elif (fitvec[sel[1]] < fitvec[sel[0]]) & \
                    (fitvec[sel[1]] < tmp_fitness):
                fitvec[sel[1]] = tmp_fitness
                popmatrix[sel[1], :] = child2

        # stopping criterion
        fitmatrix[:, i + 1] = fitvec
        ev = fitvec.max() - best_fitness
        best_fitness = fitvec.max()
        best_sol = popmatrix[_np.where(fitvec == best_fitness)[0], :]
        if ev < tol:
            stagnation += 1
            temperature = _np.maximum(0.05, _np.minimum(temperature + 0.001,
                                                        (1 - i / max_iter) ** 2))
        else:
            stagnation = 0
            temperature = 0.05
        if stagnation >= stopping:
            if verbose:
                print("\nTerminating training at iteration " + str(i+1))
            break
        elif verbose:
            print("\rIteration: " + str(i + 1) + " | Best fitness: " +
                  str(fitvec.max()) + " | Improvement <" + str(tol) + " for " +
                  str(stagnation) + "/" + str(stopping) +
                  " iterations        ", sep="", end="")
    fitmatrix = fitmatrix[:, slice(i)]
    
    # output
    for i in range(popsize):
        popmatrix[i, :] = popmatrix[i, :]*(maxval - minval) + minval
    best_sol = best_sol*(maxval - minval) + minval
    out = {"best_sol": best_sol[0],
           "best_fitness": best_fitness,
           "last_pop": popmatrix,
           "evolution": fitmatrix}
    return out
