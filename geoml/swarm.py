# geoML - machine learning models for geospatial data
# Copyright (C) 2020  Ítalo Gomes Gonçalves
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


class Particle:
    def __init__(self, size):
        self.position = _np.random.uniform(size=size)
        self.velocity = _np.random.uniform(size=size)*0.01
        self.fitness = None
        self.best_position = self.position.copy()
        self.best_fitness = None
        self.position_log = []
        self.fitness_log = []
        self.inertia_weight = 1.0


class ParticleSwarmOptimizer:
    def __init__(self, dimension, pop_size=20,
                 cognitive_constant=2,
                 social_constant=2,
                 initial_inertia=1.0,
                 final_inertia=0.1,
                 max_velocity=0.1,
                 verbose=True):
        self.dimension = dimension
        self.population = [Particle(dimension) for _ in range(pop_size)]
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.max_inertia = initial_inertia
        self.min_inertia = final_inertia
        self.max_velocity = max_velocity
        self.best_position = None
        self.best_fitness = None
        self.swarm_evolution = []
        self.fitness_evolution = []
        self.verbose = verbose

    def kickstart(self, fitness, start=None):
        if start is not None:
            self.population[0].position = start.copy()

        for i, particle in enumerate(self.population):
            if self.verbose:
                print("\rInitializing population: " + str(i+1) + "       ",
                      end="")

            fit = fitness(particle.position)
            particle.fitness = fit
            particle.position_log.append(particle.position)
            particle.fitness_log.append(fit)
            particle.best_fitness = fit

        fits = _np.array([p.best_fitness for p in self.population])
        self.best_fitness = _np.max(fits)
        self.best_position = self.population[int(_np.argmax(fits))]\
            .position.copy()

    def optimize(self, fitness, max_iter=100, patience=None, seed=None,
                 start=None, tol=0.1):

        if patience is None:
            patience = _np.maximum(50, _np.ceil(max_iter / 5))

        if seed is not None:
            _np.random.seed(seed)

        self.kickstart(fitness, start)

        stagnation = 0
        for it in range(max_iter):
            current_best = self.best_fitness

            current_inertia = (self.max_inertia
                               - (self.max_inertia
                                  - self.min_inertia) * it / max_iter)
            # main loop
            for i, particle in enumerate(self.population):
                rnd_1 = _np.random.uniform(size=len(particle.position))
                rnd_2 = _np.random.uniform(size=len(particle.position))

                inertia = particle.velocity * current_inertia
                cognitive = self.cognitive_constant * rnd_1 \
                            * (particle.best_position - particle.position)
                social = self.social_constant * rnd_2 * \
                         (self.best_position - particle.position)
                velocity = inertia + cognitive + social
                abs_vel = _np.sqrt(_np.sum(velocity ** 2))
                max_vel = self.max_velocity * _np.sqrt(self.dimension)
                if abs_vel > max_vel:
                    velocity = velocity / abs_vel * max_vel
                particle.velocity = velocity
                particle.position += velocity

                # boundaries
                particle.position = _np.minimum(particle.position,
                                                _np.ones_like(
                                                    particle.position))
                particle.position = _np.maximum(particle.position,
                                                _np.zeros_like(
                                                    particle.position))

                fit = fitness(particle.position)
                particle.fitness = fit
                particle.position_log.append(particle.position)
                particle.fitness_log.append(fit)
                if fit > particle.best_fitness:
                    particle.best_fitness = fit
                    particle.best_position = particle.position.copy()
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_position = particle.position.copy()

                if self.verbose:
                    print("\rIteration " + str(it+1) + " | Particle "
                          + str(i + 1) + " | Best fitness: "
                          + str(self.best_fitness) + "        ",
                          sep="", end="")

            # stopping criterion
            if self.best_fitness - current_best >= tol:
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                if self.verbose:
                    print("\nTerminating training at iteration "
                          + str(it + 1), end="")
                break

        if self.verbose:
            print("\n")


class SelfRegulatingPSO(ParticleSwarmOptimizer):
    def optimize(self, fitness, max_iter=100, patience=None, seed=None,
                 start=None, tol=0.1):

        if patience is None:
            patience = _np.maximum(50, _np.ceil(max_iter / 5))

        if seed is not None:
            _np.random.seed(seed)

        self.kickstart(fitness, start)

        stagnation = 0
        delta_inertia = (self.max_inertia - self.min_inertia) / max_iter
        for it in range(max_iter):
            current_best = self.best_fitness

            best_particle = _np.argmax([particle.fitness
                                        for particle in self.population])
            # main loop
            for i, particle in enumerate(self.population):
                if (i == best_particle) & (particle.fitness == current_best):
                    particle.inertia_weight += delta_inertia
                    velocity = (1 + particle.inertia_weight)*particle.velocity
                else:
                    rnd_1 = _np.random.uniform(size=len(particle.position))
                    rnd_2 = _np.random.uniform(size=len(particle.position))
                    rnd_3 = _np.random.uniform(size=len(particle.position))
                    noise = _np.random.normal(size=len(particle.position))*1e-5

                    particle.inertia_weight -= delta_inertia
                    inertia = particle.velocity * particle.inertia_weight

                    cognitive = self.cognitive_constant * rnd_1 \
                                * (particle.best_position - particle.position)
                    social = self.social_constant * rnd_2 * \
                             (self.best_position - particle.position)
                    social = _np.where(rnd_3 > 0.5,
                                       _np.zeros_like(social),
                                       social)

                    velocity = inertia + cognitive + social + noise

                velocity = _np.where(_np.abs(velocity) > self.max_velocity,
                                     _np.sign(velocity)*self.max_velocity,
                                     velocity)
                particle.velocity = velocity
                particle.position += velocity

                # boundaries
                particle.position = _np.minimum(particle.position,
                                                _np.ones_like(
                                                    particle.position))
                particle.position = _np.maximum(particle.position,
                                                _np.zeros_like(
                                                    particle.position))

                fit = fitness(particle.position)
                particle.fitness = fit
                particle.position_log.append(particle.position)
                particle.fitness_log.append(fit)
                if fit > particle.best_fitness:
                    particle.best_fitness = fit
                    particle.best_position = particle.position.copy()
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_position = particle.position.copy()

                if self.verbose:
                    print("\rIteration " + str(it+1) + " | Particle "
                          + str(i + 1) + " | Best fitness: "
                          + str(self.best_fitness) + "        ",
                          sep="", end="")

            # stopping criterion
            if self.best_fitness - current_best >= tol:
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                if self.verbose:
                    print("\nTerminating training at iteration "
                          + str(it + 1), end="")
                break

        if self.verbose:
            print("\n")
