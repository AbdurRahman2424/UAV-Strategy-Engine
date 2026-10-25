#!/usr/bin/env python3
"""Run the mission demonstration or reproducible sensitivity experiments."""
import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import numpy as np
from config import EnvironmentConfig,UAVConfig,GAConfig,FitnessWeights
from src.environment import Environment
from src.experiments import run_pair,run_experiments,result_row
from src.io_utils import write_json,write_csv
from src.visualization import generate_plots

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population',type=int,default=60)
    p.add_argument('--generations',type=int,default=100)
    p.add_argument('--mutation',type=float,default=.15)
    p.add_argument('--crossover',type=float,default=.85)
    p.add_argument('--tournament',type=int,default=3)
    p.add_argument('--elites',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--environment-seed',type=int,default=2026)
    p.add_argument('--size',type=int,default=50)
    p.add_argument('--uavs',type=int,default=3)
    p.add_argument('--waypoints',type=int,default=4)
    p.add_argument('--patience',type=int,default=0,help='Stagnation generations; 0 disables early stopping')
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--experiments',action='store_true')
    p.add_argument('--experiment-budget',type=int,default=600)
    p.add_argument('--comparison-budget',type=int,default=1800)
    p.add_argument('--experiment-seeds',type=int,nargs='+',default=[42,43,44])
    return p


def main():
    args = parser().parse_args()
    ec = EnvironmentConfig(size=args.size,seed=args.environment_seed)
    spec = UAVConfig(count=args.uavs,waypoints=args.waypoints)
    config = GAConfig(population=args.population,generations=args.generations,mutation=args.mutation,
                      crossover=args.crossover,tournament=args.tournament,elites=args.elites,
                      seed=args.seed,patience=args.patience)
    env = Environment(ec)
    output = args.output
    for folder in ('figures','results','screenshots'):
        (output/folder).mkdir(parents=True,exist_ok=True)
    if args.experiments:
        run_experiments(env,spec,output,args.experiment_budget,tuple(args.experiment_seeds),args.comparison_budget)
        return
    print(f'Mission: {ec.size}x{ec.size}, {spec.count} UAVs, environment seed {ec.seed}, optimizer seed {config.seed}',flush=True)
    ga,random = run_pair(env,spec,config,progress=lambda g,n,f:print(f'Generation {g:3d} | evaluations {n:5d} | best {f:.6f}',flush=True))
    for name,result in (('GA',ga),('Random Search',random)):
        if not result.evaluation.feasible:
            raise RuntimeError(f'{name} returned an infeasible solution.')
        value = result.evaluation
        print(f'{name}: fitness={value.fitness:.6f}, weighted coverage={value.coverage:.2%}, detection={value.detection:.2%}, '
              f'energy={value.energy:.3f}, time={value.mission_time:.3f}s, risk={value.risk:.3f} hazard-s, '
              f'valid={result.valid_solutions}/{result.evaluations}, runtime={result.seconds:.2f}s',flush=True)
        write_json(output/'results'/f'{"ga" if name=="GA" else "random"}_solution.json',
                   dict(chromosome=result.best.to_dict(),metrics=value.summary(),paths=result.evaluation.paths))
        np.savez_compressed(output/'results'/f'{"ga" if name=="GA" else "random"}_coverage.npz',
                            observation_counts=value.observation_counts,detection_probability=value.detection_map)
    env.coverage = ga.evaluation.observation_counts>0
    env.save(output/'results'/'environment.npz')
    metadata = dict(environment=asdict(ec),uav=asdict(spec),ga_config=asdict(config),weights=asdict(FitnessWeights()),
                    python=platform.python_version(),numpy=np.__version__,ga=ga.evaluation.summary(),
                    random=random.evaluation.summary(),ga_evaluations=ga.evaluations,random_evaluations=random.evaluations,
                    ga_seconds=ga.seconds,random_seconds=random.seconds,
                    initial_ga_fitness=ga.history[0]['best_fitness'],
                    final_generation=ga.history[-1]['generation'],
                    ga_valid_solutions=ga.valid_solutions,random_valid_solutions=random.valid_solutions)
    write_json(output/'results'/'run_summary.json',metadata)
    write_csv(output/'results'/'ga_history.csv',ga.history)
    write_csv(output/'results'/'random_history.csv',random.history)
    write_json(output/'results'/'evolution_snapshots.json',
               {str(g):dict(chromosome=c.to_dict(),metrics=v.summary(),paths=v.paths)
                for g,(c,v) in ga.snapshots.items()})
    write_csv(output/'results'/'main_comparison.csv',[result_row(ga,'GA',config.seed),result_row(random,'Random Search',config.seed)])
    generate_plots(env,ga,random,output/'figures')
    print(f'Saved numerical results and figures to {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
