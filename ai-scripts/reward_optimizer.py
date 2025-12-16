# gpt all the way. this script is not used really
import random
import copy
from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod


class RewardGenome:
    def __init__(self, reward_dict: Dict[str, float]):
        self.rewards = reward_dict
        self.fitness: float = 0.0  # Will be set after simulation

    def mutate(self, mutation_rate=0.1, mutation_scale=0.5):
        """Apply random mutations to reward values."""
        new_rewards = copy.deepcopy(self.rewards)
        for key in new_rewards:
            if random.random() < mutation_rate:
                change = random.uniform(-mutation_scale, mutation_scale)
                new_rewards[key] += change
        return RewardGenome(new_rewards)

    def crossover(self, other):
        """Combine with another genome to produce a child genome."""
        child_rewards = {}
        for key in self.rewards:
            child_rewards[key] = random.choice([self.rewards[key], other.rewards[key]])
        return RewardGenome(child_rewards)

    def __repr__(self):
        return f"RewardGenome(fitness={self.fitness}, rewards={self.rewards})"


class SimulatorInterface(ABC):
    @abstractmethod
    def simulate(self, genome: RewardGenome) -> float:
        """
        Run the game simulation using the provided reward genome.
        Should return a fitness value based on average fitness growth.
        """
        pass


class EvolutionaryOptimizer:
    def __init__(self, simulator: SimulatorInterface, base_rewards: Dict[str, float], population_size=10):
        self.simulator = simulator
        self.population_size = population_size
        self.population: List[RewardGenome] = [
            RewardGenome(self._mutate_base(base_rewards, scale=1.0)) for _ in range(population_size)
        ]
        self.base_rewards = base_rewards

    def _mutate_base(self, rewards, scale=0.5):
        """Helper to create varied initial population."""
        return {k: v + random.uniform(-scale, scale) for k, v in rewards.items()}

    def run(self, generations=10, top_k=3, mutation_rate=0.2, mutation_scale=0.3):
        for gen in range(generations):
            print(f"\n--- Generation {gen + 1} ---")

            # Evaluate all genomes
            for genome in self.population:
                if genome.fitness is None:
                    genome.fitness = self.simulator.simulate(genome)
                    print(f"Evaluated: {genome}")

            # Select top genomes
            sorted_population = sorted(self.population, key=lambda g: g.fitness, reverse=True)
            elites = sorted_population[:top_k]

            print(f"Top {top_k} genomes:")
            for elite in elites:
                print(elite)

            # Reproduce
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(elites, 2)
                child = parent1.crossover(parent2)
                mutated_child = child.mutate(mutation_rate, mutation_scale)
                new_population.append(mutated_child)

            self.population = new_population

        # Return the best genome after all generations
        best = max(self.population, key=lambda g: g.fitness)
        print(f"\nBest reward config found: {best}")
        return best


if __name__ == "__main__":
    base_rewards = {
        "TOOK_DAMAGE": -1.0,
        "TIME_ALIVE": 0.05,
        "HIT_PLAYER": 15.0,
        "RETREATED": -2.0,
        "WASTED_MOVEMENT": -1.0,
        "MOVED_CLOSER": 1.0,
        "MISSED": 0.0,
    }

    # simulator = SimulatorInterface()  # Replace with actual simulator implementation
    # optimizer = EvolutionaryOptimizer(simulator, base_rewards, population_size=8)
    # best_genome = optimizer.run(generations=5, top_k=2)