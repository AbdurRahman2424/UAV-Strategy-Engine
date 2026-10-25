"""Configurable tournament selection (maximize scalar fitness)."""


def tournament_select(population, evaluations, rng, size=3):
    indices = rng.choice(len(population),size=size,replace=False)
    best = max(indices,key=lambda i:evaluations[i].fitness)
    return population[best]
