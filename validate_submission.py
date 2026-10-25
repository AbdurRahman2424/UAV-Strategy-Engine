#!/usr/bin/env python3
"""Independently validate saved results and report; fail loudly on inconsistencies."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from config import EnvironmentConfig,UAVConfig
from src.chromosome import Chromosome
from src.drone import UAVPlan
from src.environment import Environment
from src.fitness import FitnessEvaluator
from src.pathfinding import AStarPlanner
from src.io_utils import write_json

ROOT = Path(__file__).resolve().parent


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def read_csv(file):
    with file.open() as stream:
        return list(csv.DictReader(stream))


def validate(output=ROOT/'outputs',report=ROOT/'report'/'Technical_Report.pdf'):
    output,report = Path(output),Path(report)
    results = output/'results'
    run = json.loads((results/'run_summary.json').read_text())
    env = Environment(EnvironmentConfig(**run['environment']))
    settings = dict(run['uav'])
    settings['power_levels'] = tuple(settings['power_levels'])
    spec = UAVConfig(**settings)
    evidence = {'checks':[]}
    with np.load(results/'environment.npz') as saved:
        for key in ('priority','hazard','traversable','no_fly','obstacles','targets','target_probability'):
            require(np.array_equal(saved[key],getattr(env,key)),f'Environment field mismatch: {key}')
    evidence['checks'].append('Saved environment exactly matches configured seed')
    for name in ('ga','random'):
        saved = json.loads((results/f'{name}_solution.json').read_text())
        plans = []
        for raw in saved['chromosome']['uavs']:
            raw['start'] = tuple(raw['start'])
            raw['waypoints'] = list(map(tuple,raw['waypoints']))
            plans.append(UAVPlan(**raw))
        chromosome = Chromosome(plans)
        value = FitnessEvaluator(env,spec,AStarPlanner(env)).evaluate(chromosome)
        require(value.feasible,f'{name} saved mission is infeasible: {value.violations}')
        actual = [[list(p) for p in path] for path in value.paths]
        require(actual==saved['paths'],f'{name} stored expanded paths do not match encoded route')
        for metric in ('fitness','coverage','detection','energy','mission_time','risk','expected_detections'):
            require(np.isclose(getattr(value,metric),saved['metrics'][metric],rtol=1e-10,atol=1e-10),
                    f'{name} mismatched recomputed metric {metric}')
            require(saved['metrics'][metric]==run[name][metric],f'{name} summary disagrees with solution JSON')
        with np.load(results/f'{name}_coverage.npz') as coverage:
            require(np.array_equal(coverage['observation_counts'],value.observation_counts),f'{name} coverage mismatch')
            require(np.allclose(coverage['detection_probability'],value.detection_map),f'{name} detection mismatch')
        for path in value.paths:
            require(all(env.valid(p) and not env.no_fly[p[1],p[0]] for p in path),f'{name}: blocked route')
            require(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:])),f'{name}: disconnected route')
        for uav in value.uav_metrics:
            require(uav['distance']<=spec.max_range+1e-9 and uav['energy']<=spec.battery_capacity+1e-9,
                    f'{name}: UAV resource limit exceeded')
        evidence['checks'].append(f'{name}: saved chromosome, expanded paths, all constraints and numerical metrics independently verified')
    require(run['ga_evaluations']==run['random_evaluations'],'Unequal main optimizer budgets')
    ga_history = read_csv(results/'ga_history.csv')
    random_history = read_csv(results/'random_history.csv')
    ga_values = [float(row['best_fitness']) for row in ga_history]
    require(np.all(np.diff(ga_values)>=-1e-12),'GA best-so-far fitness decreased')
    require(ga_values[-1]>ga_values[0],'GA failed to improve from its initial population')
    require(np.all(np.diff([float(r['best_fitness']) for r in random_history])>=-1e-12),'Random best-so-far decreased')
    require([int(h['evaluations']) for h in ga_history]==[int(h['evaluations']) for h in random_history],
            'GA and baseline convergence checkpoints do not match')
    require(int(ga_history[-1]['evaluations'])==run['ga_evaluations'],'History budget does not match summary')
    require(np.isclose(ga_values[-1],run['ga']['fitness']),'History final best disagrees with summary')
    evidence['checks'].append('Actual fitness improvement, monotone best histories and equal main evaluation budgets verified')
    experiments = json.loads((results/'experiment_summary.json').read_text())
    require(experiments['environment_seed']==env.config.seed and experiments['grid_size']==env.size,
            'Experiment and main-run environments differ')
    rows = read_csv(results/'sensitivity.csv')
    seeds = experiments['seeds']
    require(len(seeds)>=3,'At least three experimental seeds required for the submitted assignment')
    expected = {(family,float(v),seed) for family,values in
                {'population':[20,40,60,100],'mutation':[.05,.10,.15,.30],'tournament':[2,3,5,7]}.items()
                for v in values for seed in seeds}
    actual = {(r['parameter'],float(r['value']),int(r['seed'])) for r in rows}
    require(actual==expected and len(rows)==len(expected),'Missing or duplicate sensitivity runs')
    require(all(int(r['evaluations'])==experiments['sensitivity_evaluation_budget'] for r in rows),
            'Sensitivity budgets differ')
    require(all(float(r['feasible_fraction'])==1 for r in rows),'Infeasible evaluated experimental candidates')
    histories = read_csv(results/'sensitivity_histories.csv')
    for row in histories:
        expected_size = int(float(row['value'])) if row['parameter']=='population' else 60
        require(int(row['population_size'])==expected_size,'Partial generation changed population size')
    for family,means in experiments['sensitivity'].items():
        for value,mean in means.items():
            measured = [float(r['fitness']) for r in rows if r['parameter']==family and float(r['value'])==float(value)]
            require(np.isclose(np.mean(measured),mean),'Sensitivity summary mean differs from measured CSV')
    paired = read_csv(results/'paired_comparison.csv')
    require(len(paired)==2*len(seeds),'Incomplete independent comparisons')
    for seed in seeds:
        group = [r for r in paired if int(r['seed'])==seed]
        require({r['algorithm'] for r in group}=={'GA','Random Search'},f'Incomplete comparison seed {seed}')
        require(all(int(r['evaluations'])==experiments['comparison_evaluation_budget'] for r in group),
                'Independent comparison budgets differ')
    evidence['checks'].append(f'{len(rows)} sensitivity runs and {len(paired)} paired optimizer runs complete with equal budgets')
    figures = ['fitness_convergence','optimized_flight_paths','coverage_heatmap',
               'evolution_across_generations','ga_vs_random_search','hyperparameter_sensitivity']
    for filename in figures:
        with Image.open(output/'figures'/f'{filename}.png') as im:
            require(im.width>=700 and im.height>=500,f'Figure is unexpectedly small: {filename}')
            im.verify()
    evidence['checks'].append('All six required PNG figures exist and decode correctly')
    require(report.is_file() and report.stat().st_size>10000,'Technical report missing or empty')
    if shutil.which('pdfinfo') and shutil.which('pdftotext'):
        info = subprocess.run(['pdfinfo',str(report)],text=True,capture_output=True,check=True).stdout
        pages = int(next(line.split(':')[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
        require(pages==4,f'Expected four report pages, found {pages}')
        contents = subprocess.run(['pdftotext',str(report),'-'],text=True,capture_output=True,check=True).stdout
        for section in ('Introduction','Methodology','Results and experiments','Discussion','Team Member 1','Instructor'):
            require(section in contents,f'Report missing {section}')
        evidence['report_pages'] = pages
        evidence['checks'].append('Four-page PDF and required section/placeholder text verified with Poppler')
    else:
        evidence['checks'].append('PDF exists; install Poppler to enable page/text verification')
    evidence.update(passed=True,main_evaluations_per_optimizer=run['ga_evaluations'],
                    ga_fitness=run['ga']['fitness'],random_fitness=run['random']['fitness'],
                    sensitivity_runs=len(rows),paired_optimizer_runs=len(paired))
    write_json(results/'validation.json',evidence)
    for check in evidence['checks']:
        print(f'PASS: {check}')
    return evidence


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--report',type=Path,default=ROOT/'report'/'Technical_Report.pdf')
    args = p.parse_args()
    validate(args.output,args.report)
