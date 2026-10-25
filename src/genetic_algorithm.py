"""A hand-written GA with elitism, tournament selection, and random immigrants."""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from config import GAConfig
from .chromosome import Chromosome
from .selection import tournament_select


@dataclass
class OptimizationResult:
    best: Chromosome
    evaluation: object
    history: list
    snapshots: dict = field(default_factory=dict)
    evaluations: int = 0
    valid_solutions: int = 0
    seconds: float = 0.


def population_diversity(population, env, spec):
    """Mean normalized gene standard deviation, including all heterogeneous genes."""
    vectors = []
    for chromosome in population:
        vector = []
        for p in chromosome.plans:
            vector.extend(np.array([p.start,*p.waypoints]).ravel()/(env.size-1))
            speed_span = max(spec.max_speed-spec.min_speed,1e-12)
            vector.extend((np.array(p.speeds)-spec.min_speed)/speed_span)
            vector.extend(p.sensors)
            power_span = max(max(spec.power_levels)-min(spec.power_levels),1e-12)
            vector.extend((np.array(p.sensor_power)-min(spec.power_levels))/power_span)
        vectors.append(vector)
    return float(np.std(vectors,axis=0).mean())


class GeneticAlgorithm:
    def __init__(self, environment, spec, evaluator, config=GAConfig()):
        self.env,self.spec,self.evaluator,self.config = environment,spec,evaluator,config
        self.rng = np.random.default_rng(config.seed)

    def run(self, evaluation_budget=None, progress=None):
        start = perf_counter()
        c,rng = self.config,self.rng
        if evaluation_budget is not None and evaluation_budget < c.population:
            raise ValueError('Evaluation budget must cover the initial population.')
        population = [Chromosome.random(self.env,self.spec,self.evaluator.planner,rng)
                      for _ in range(c.population)]
        values = [self.evaluator.evaluate(p) for p in population]
        count,valid = len(values),sum(v.feasible for v in values)
        result = OptimizationResult(population[0].copy(),values[0],[])
        last_improvement,best_fitness = 0,-float('inf')
        snapshot_generations = {0, min(10,c.generations), min(25,c.generations),
                                min(50,c.generations),c.generations}
        # Short runs still show multiple distinct stages.
        if c.generations < 50:
            snapshot_generations.update(np.linspace(0,c.generations,min(5,c.generations+1),dtype=int))
        for generation in range(c.generations+1):
            order = sorted(range(len(population)),key=lambda i:values[i].fitness,reverse=True)
            population,values = [population[i] for i in order],[values[i] for i in order]
            if values[0].fitness > best_fitness+1e-12:
                best_fitness = values[0].fitness
                last_improvement = generation
                result.best,result.evaluation = population[0].copy(),values[0]
            diversity = population_diversity(population,self.env,self.spec)
            result.history.append(dict(generation=generation,evaluations=count,best_fitness=best_fitness,
                                       average_fitness=float(np.mean([v.fitness for v in values])),
                                       population_size=len(population),diversity=diversity,
                                       feasible_fraction=float(np.mean([v.feasible for v in values]))))
            if generation in snapshot_generations:
                result.snapshots[generation] = (result.best.copy(),result.evaluation)
            if progress and (generation%10 == 0 or generation == c.generations):
                progress(generation,count,best_fitness)
            if generation == c.generations or (c.patience and generation-last_improvement>=c.patience):
                break
            if evaluation_budget is not None and count >= evaluation_budget:
                break
            new_pop,new_vals = population[:c.elites],values[:c.elites]
            children_count = c.population-c.elites
            if evaluation_budget is not None:
                children_count = min(children_count,evaluation_budget-count)
            # Diversity collapse triggers more immigrants without changing the budget.
            immigrant_rate = max(c.immigrant_fraction,.15 if diversity < .06 else 0.)
            immigrants = min(children_count,int(round(c.population*immigrant_rate)))
            for j in range(children_count):
                if j < immigrants:
                    child = Chromosome.random(self.env,self.spec,self.evaluator.planner,rng)
                else:
                    a = tournament_select(population,values,rng,c.tournament)
                    if rng.random() < c.crossover:
                        b = tournament_select(population,values,rng,c.tournament)
                        child = a.crossover(b,self.env,self.spec,self.evaluator.planner,rng,
                                            'segment' if rng.random()<.5 else 'geometric')
                    else:
                        child = a.copy()
                    child = child.mutate(self.env,self.spec,self.evaluator.planner,rng,c.mutation)
                value = self.evaluator.evaluate(child)
                new_pop.append(child)
                new_vals.append(value)
                count += 1
                valid += value.feasible
            # Keep population size fixed when the budget permits a partial generation.
            missing = c.population-len(new_pop)
            if missing:
                new_pop.extend(population[c.elites:c.elites+missing])
                new_vals.extend(values[c.elites:c.elites+missing])
            population,values = new_pop,new_vals
        result.snapshots[result.history[-1]['generation']] = (result.best.copy(),result.evaluation)
        result.evaluations,result.valid_solutions,result.seconds = count,valid,perf_counter()-start
        return result
