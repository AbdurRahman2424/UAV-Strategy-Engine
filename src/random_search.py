"""Independent random-search baseline using the identical repair and fitness."""
from time import perf_counter
import numpy as np
from .chromosome import Chromosome
from .genetic_algorithm import OptimizationResult


class RandomSearch:
    def __init__(self, environment, spec, evaluator, seed=10042):
        self.env,self.spec,self.evaluator = environment,spec,evaluator
        self.rng = np.random.default_rng(seed)

    def run(self,evaluation_budget,checkpoints=None):
        if evaluation_budget < 1:
            raise ValueError('Random search requires a positive evaluation budget.')
        start = perf_counter()
        checkpoints = set(checkpoints or range(1,evaluation_budget+1)) | {evaluation_budget}
        result = OptimizationResult(None,None,[])
        best = -float('inf')
        valid = 0
        for count in range(1,evaluation_budget+1):
            chromosome = Chromosome.random(self.env,self.spec,self.evaluator.planner,self.rng)
            value = self.evaluator.evaluate(chromosome)
            valid += value.feasible
            if value.fitness > best:
                best = value.fitness
                result.best,result.evaluation = chromosome,value
            if count in checkpoints:
                result.history.append(dict(evaluations=count,best_fitness=best))
        result.evaluations,result.valid_solutions,result.seconds = evaluation_budget,valid,perf_counter()-start
        return result
