"""Equal-budget sensitivity sweeps and repeated GA/random-search comparisons."""
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
from config import GAConfig
from .pathfinding import AStarPlanner
from .fitness import FitnessEvaluator
from .genetic_algorithm import GeneticAlgorithm
from .random_search import RandomSearch
from .io_utils import write_csv,write_json
from .visualization import sensitivity_plot


def run_pair(env,spec,config,budget=None,progress=None):
    ga = GeneticAlgorithm(env,spec,FitnessEvaluator(env,spec,AStarPlanner(env)),config).run(budget,progress)
    random = RandomSearch(env,spec,FitnessEvaluator(env,spec,AStarPlanner(env)),
                          seed=config.seed+10000).run(ga.evaluations,[h['evaluations'] for h in ga.history])
    return ga,random


def result_row(result,algorithm,seed):
    row = dict(algorithm=algorithm,seed=seed,evaluations=result.evaluations,
               valid_solutions=result.valid_solutions,seconds=result.seconds)
    row.update({k:v for k,v in result.evaluation.summary().items() if k not in ('violations','uav_metrics')})
    return row


def run_experiments(env,spec,output,budget=600,seeds=(42,43,44),comparison_budget=1800,log=print):
    """Each sensitivity run has exactly `budget` evaluator calls, including initialization.

    Last generations may be partial so different populations receive equal budgets.
    Repeated comparison is separate from sensitivity and uses its own common budget.
    """
    if budget < 100 or comparison_budget < 60 or not seeds:
        raise ValueError('Sensitivity budget >= 100, comparison budget >= 60, and seeds required.')
    output = Path(output)
    rows,histories = [],[]
    variants = {'population':[20,40,60,100],'mutation':[.05,.10,.15,.30],'tournament':[2,3,5,7]}
    for parameter,values in variants.items():
        for value in values:
            for seed in seeds:
                settings = {parameter:value,'seed':seed}
                c = replace(GAConfig(),**settings)
                c = replace(c,generations=math.ceil((budget-c.population)/(c.population-c.elites)))
                evaluator = FitnessEvaluator(env,spec,AStarPlanner(env))
                result = GeneticAlgorithm(env,spec,evaluator,c).run(budget)
                first,final = result.history[0]['best_fitness'],result.evaluation.fitness
                threshold = first+.9*(final-first)
                convergence = next(h['evaluations'] for h in result.history if h['best_fitness']>=threshold-1e-12)
                rows.append(dict(parameter=parameter,value=value,seed=seed,evaluations=result.evaluations,
                                 generations=result.history[-1]['generation'],fitness=final,
                                 coverage=result.evaluation.coverage,detection=result.evaluation.detection,
                                 energy=result.evaluation.energy,mission_time=result.evaluation.mission_time,
                                 risk=result.evaluation.risk,convergence_evaluations=convergence,
                                 diversity=result.history[-1]['diversity'],
                                 feasible_fraction=result.valid_solutions/result.evaluations,seconds=result.seconds))
                histories.extend(dict(parameter=parameter,value=value,seed=seed,**h) for h in result.history)
                log(f'Sensitivity {parameter}={value}, seed={seed}: F={final:.4f}, n={result.evaluations}, {result.seconds:.1f}s')
    write_csv(output/'results'/'sensitivity.csv',rows)
    write_csv(output/'results'/'sensitivity_histories.csv',histories)
    sensitivity_plot(rows,output/'figures')
    comparison = []
    for seed in seeds:
        config = replace(GAConfig(),seed=seed,generations=math.ceil((comparison_budget-60)/58))
        ga,random = run_pair(env,spec,config,comparison_budget)
        comparison.extend([result_row(ga,'GA',seed),result_row(random,'Random Search',seed)])
        write_csv(output/'results'/f'comparison_ga_history_seed_{seed}.csv',ga.history)
        write_csv(output/'results'/f'comparison_random_history_seed_{seed}.csv',random.history)
        log(f'Paired seed={seed}: GA={ga.evaluation.fitness:.4f}, Random={random.evaluation.fitness:.4f}, n={ga.evaluations}')
    write_csv(output/'results'/'paired_comparison.csv',comparison)
    summary = {'environment_seed':env.config.seed,'grid_size':env.size,'sensitivity_evaluation_budget':budget,
               'comparison_evaluation_budget':comparison_budget,'seeds':list(seeds),
               'budget_policy':'Equal fitness evaluations, including initialization; elites are not re-evaluated; final generation may be partial.',
               'number_of_sensitivity_runs':len(rows),'number_of_comparison_runs':len(comparison),
               'sensitivity':{},'comparison':{}}
    for parameter in variants:
        group = [r for r in rows if r['parameter']==parameter]
        means = {str(v):float(np.mean([r['fitness'] for r in group if r['value']==v])) for v in variants[parameter]}
        summary['sensitivity'][parameter] = means
    for algo in ('GA','Random Search'):
        group = [r for r in comparison if r['algorithm']==algo]
        summary['comparison'][algo] = {metric:{'mean':float(np.mean([r[metric] for r in group])),
                                                  'std':float(np.std([r[metric] for r in group],ddof=1)) if len(group)>1 else 0.}
                                            for metric in ('fitness','coverage','detection','energy','mission_time','risk','seconds')}
    write_json(output/'results'/'experiment_summary.json',summary)
    return summary
