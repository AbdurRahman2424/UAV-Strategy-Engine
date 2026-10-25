#!/usr/bin/env python3
"""Create the four-page technical report from saved measurements, never dummy data."""
import argparse
import csv
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

ROOT = Path(__file__).resolve().parent


def generate(output=ROOT/'outputs',destination=ROOT/'report'/'Technical_Report.pdf'):
    output,destination = Path(output),Path(destination)
    required = [output/'results'/'run_summary.json',output/'results'/'experiment_summary.json',
                output/'results'/'sensitivity.csv']
    for file in required:
        if not file.is_file():
            raise FileNotFoundError(f'{file} is missing. Run main.py and main.py --experiments first.')
    run = json.loads(required[0].read_text())
    experiments = json.loads(required[1].read_text())
    if experiments['environment_seed'] != run['environment']['seed'] or experiments['grid_size'] != run['environment']['size']:
        raise ValueError('Report inputs use different environments; rerun both commands with matching settings.')
    with required[2].open() as stream:
        sensitivity = list(csv.DictReader(stream))
    font_dir = Path('/usr/share/fonts/truetype/dejavu')
    if (font_dir/'DejaVuSans.ttf').exists():
        pdfmetrics.registerFont(TTFont('ReportSans',str(font_dir/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('ReportSans-Bold',str(font_dir/'DejaVuSans-Bold.ttf')))
        pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold',italic='ReportSans',boldItalic='ReportSans-Bold')
        regular,bold = 'ReportSans','ReportSans-Bold'
    else:
        regular,bold = 'Helvetica','Helvetica-Bold'
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('Body',fontName=regular,fontSize=9,leading=13,spaceAfter=7,textColor=colors.HexColor('#253447')))
    styles.add(ParagraphStyle('SmallBody',parent=styles['Body'],fontSize=8,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle('Section',fontName=bold,fontSize=12,leading=15,spaceBefore=9,spaceAfter=7,textColor=colors.HexColor('#174a70')))
    styles.add(ParagraphStyle('TitleCustom',fontName=bold,fontSize=22,leading=28,spaceAfter=12,textColor=colors.HexColor('#123b56')))
    styles.add(ParagraphStyle('CaptionCustom',parent=styles['SmallBody'],fontSize=7.5,leading=10,alignment=TA_CENTER,textColor=colors.HexColor('#536475')))
    story = []
    width = A4[0]-88

    def p(text,style='Body'):
        story.append(Paragraph(text,styles[style]))

    def heading(text):
        p(text,'Section')

    def table(data,widths=None):
        formatted = [[Paragraph(str(cell),styles['SmallBody']) for cell in row] for row in data]
        t = Table(formatted,colWidths=widths or [width/len(data[0])]*len(data[0]),hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf3')),
                               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f8')]),
                               ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                               ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),
                               ('BOTTOMPADDING',(0,0),(-1,-1),3),
                               ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#a8c0d0'))]))
        story.append(t)
        story.append(Spacer(1,7))

    def figure(name,max_width,max_height,caption):
        file = output/'figures'/name
        if not file.is_file():
            raise FileNotFoundError(file)
        im = Image(str(file))
        scale = min(max_width/im.imageWidth,max_height/im.imageHeight)
        im.drawWidth,im.drawHeight = im.imageWidth*scale,im.imageHeight*scale
        story.append(im)
        p(caption,'CaptionCustom')

    ga,rs,spec,cfg = run['ga'],run['random'],run['uav'],run['ga_config']
    env = run['environment']
    p('Multi-Objective UAV<br/>Strategy Engine Using<br/>Genetic Algorithms','TitleCustom')
    p('UNIVERSITY ARTIFICIAL INTELLIGENCE ASSIGNMENT · TECHNICAL REPORT','SmallBody')
    table([['Submission details','Team of three'],
           ['University: ____________________<br/>Course: ________________________<br/>Instructor: _____________________',
            'Team Member 1: __________________<br/>Team Member 2: __________________<br/>Team Member 3: __________________']],
          [width/2,width/2])
    heading('1. Introduction and problem formulation')
    p('Disaster response requires a UAV fleet to survey important areas while conserving energy and avoiding hazardous or prohibited terrain. '
      'This project implements a reproducible, locally executable simulation and a hand-written Genetic Algorithm (GA). '
      'A strategy jointly chooses deployment sites, waypoint trajectories, segment speeds, sensor activation and discrete sensor power.')
    p('The mission maximizes priority-weighted unique coverage and expected target detection, while minimizing fleet energy, completion time and hazard exposure. '
      'Geometry and sensor decisions combine discrete, binary and continuous variables. Nonlinear sensing and shared coverage make strategies interdependent; '
      'a GA can explore these combinations without requiring gradients. The result is a weighted compromise, rather than a proof of global optimality or a Pareto front.')
    heading('Environment and physical assumptions')
    p(f'The {env["size"]} × {env["size"]} grid uses environment seed {env["seed"]}. Each cell stores traversability, priority (1, 3 or 5), '
      'hazard intensity in [0, 1], target probability, sampled target presence and coverage. Circular priority zones, rectangular buildings and no-fly zones, '
      f'and smooth hazard regions define the terrain. {env["target_count"]} simulated targets are sampled without replacement from a priority-biased probability map. '
      'Optimization uses this prior; target locations provide a separate expected-detections diagnostic. A* uses four-neighbour movement and hazard-weighted edges.')
    table([['UAV specification','Configured value'],
           ['Fleet / waypoints',f'{spec["count"]} UAVs / {spec["waypoints"]} encoded segments per UAV'],
           ['Range / battery',f'{spec["max_range"]:g} cells / {spec["battery_capacity"]:g} model energy units per UAV'],
           ['Speed / sensing',f'{spec["min_speed"]:g}–{spec["max_speed"]:g} cells/s; base radius {spec["sensor_radius"]:g} cells'],
           ['Power levels / sensor draw',f'{spec["power_levels"]}; {spec["sensor_consumption"]:g} × power energy units/s']],
          [width*.42,width*.58])
    p('Missions are one-way surveys from externally supplied deployment sites; launch logistics and return-to-base are excluded. '
      'A cell is one abstract distance unit. Energy is a transparent surrogate model, not a calibrated hardware battery specification. '
      'All hard constraints are verified for the expanded grid route, not just its encoded waypoints.','SmallBody')
    story.append(PageBreak())

    heading('2. Methodology and mathematical model')
    p('<b>Hybrid chromosome.</b> Each fleet chromosome contains one UAVPlan per vehicle: '
      '<b>{start: (x,y), waypoints: [(x,y)]×K, speeds: [v]×K, sensors: [0/1]×K, sensor_power: [p]×K}</b>. '
      'Segment j begins at deployment (j=0) or the previous waypoint. Speeds, sensing and power stay linked to segments during crossover. '
      'Deep copies preserve parent independence. Each waypoint pair is expanded with cached A*, whose edge cost is 1 + 1.8(h<sub>a</sub> + h<sub>b</sub>)/2.')
    heading('Normalized multi-objective fitness')
    p('<b>F = 0.35 C + 0.25 D − 0.15 E<sub>n</sub> − 0.10 T<sub>n</sub> − 0.15 R<sub>n</sub> − P</b>')
    p('Let S be traversable cells, q<sub>c</sub> their priorities, π<sub>c</sub> the normalized target prior, h<sub>c</sub> hazard intensity, '
      'and b<sub>c</sub> indicate at least one active sensor footprint. Then <b>C = Σ q<sub>c</sub>b<sub>c</sub> / Σ q<sub>c</sub></b>, '
      'with both sums over S. Reobserving a cell never adds new unique coverage. '
      '<b>D = Σ π<sub>c</sub>[1 − exp(−z<sub>c</sub>)]</b>, where z accumulates exposure from every active visit.')
    p('A visit to arrival cell a at speed v and power p has duration Δt = 1/v and radius r = r<sub>0</sub>√p. '
      'For each traversable cell c within r, add <b>0.65 p Δt exp(−d(a,c)² / (2r²))(1 − 0.4h<sub>c</sub>)</b> '
      'to z<sub>c</sub>. This independent-exposure model saturates detection probability as observations accumulate. '
      'Inactive or zero-length segments produce no observations. The sampled-target diagnostic sums these probabilities at actual target cells.')
    p('For UAV i, <b>E<sub>i</sub> = Σ L<sub>j</sub>(0.26 + 0.045v<sub>j</sub>² + 0.22s<sub>j</sub>p<sub>j</sub>/v<sub>j</sub>)</b> '
      'and <b>T<sub>i</sub> = Σ L<sub>j</sub>/v<sub>j</sub></b>; L is the number of A* grid edges and s is the sensor state. '
      '<b>R<sub>i</sub> = Σ h<sub>a</sub>/v<sub>j</sub></b> over arrival cells along the route. Thus higher speeds reduce time but increase '
      'propulsion expenditure and reduce sensing dwell. Hazards affect route selection, risk and sensing quality.')
    p('For N identical UAVs with battery B, range L<sub>max</sub> and minimum speed v<sub>min</sub>, '
      '<b>E<sub>n</sub> = Σ E<sub>i</sub>/(NB)</b>, <b>T<sub>n</sub> = max T<sub>i</sub>/(L<sub>max</sub>/v<sub>min</sub>)</b>, '
      'and <b>R<sub>n</sub> = Σ R<sub>i</sub>/(NL<sub>max</sub>/v<sub>min</sub>)</b>. '
      'All five objectives lie in [0,1] for feasible solutions. Fixed physical bounds avoid generation-dependent normalization. '
      'Coverage and detection receive 60% of weight to prioritize service; the remaining 40% controls resource and exposure costs. '
      'Weights express a policy preference and are not empirically learned.')
    heading('Evolutionary operators and hard constraints')
    p(f'Tournament selection (size {cfg["tournament"]}) chooses parents from a population of {cfg["population"]}. '
      f'Crossover probability is {cfg["crossover"]:.2f}; segment swapping and arithmetic waypoint recombination are selected equally. '
      f'Per-gene mutation probability is {cfg["mutation"]:.2f}: local spatial moves, occasional waypoint replacement, speed perturbation, sensor toggle, '
      f'power changes and deployment moves. The best {cfg["elites"]} survive; about 5% random immigrants replenish diversity, rising to 15% if mean '
      'normalized gene standard deviation falls below 0.06. Elitism preserves quality; immigrants supply exploration.')
    p('Repair projects coordinates into traversable cells within the deployment component, clamps speeds, snaps powers and shortens segments until battery/range limits hold. '
      'It replans truncated endpoints and checks the resulting A* route again. The evaluator independently checks structure, connectivity, bounds and resources. '
      'Feasible P=0; an invalid chromosome receives F=−1−V, where V is its violation count, and is never shown as a valid final mission. '
      'Workflow: initialize → repair/evaluate → select → crossover/mutate → repair/evaluate → elite survivor selection; terminate at the generation/budget limit or optional stagnation.')
    story.append(PageBreak())

    heading('3. Results and experiments')
    p(f'The main run used optimizer seed {cfg["seed"]}, {run["final_generation"]} generations and '
      f'<b>{run["ga_evaluations"]:,} fitness evaluations per optimizer</b>. Initial GA best fitness was {run["initial_ga_fitness"]:.5f}; '
      f'final fitness was {ga["fitness"]:.5f}. Random search used identical terrain, priors, initialization distribution, repair and fitness, '
      'with an independent RNG stream. Elites are not re-evaluated; every generated candidate evaluation counts.')
    figure('fitness_convergence.png',width,152,'Figure 1. Recorded fitness traces, aligned by cumulative evaluator calls; top axis shows GA generation.')
    table([['Main run metric','GA','Random search'],
           ['Best fitness',f'{ga["fitness"]:.5f}',f'{rs["fitness"]:.5f}'],
           ['Weighted coverage / detection',f'{ga["coverage"]:.1%} / {ga["detection"]:.1%}',f'{rs["coverage"]:.1%} / {rs["detection"]:.1%}'],
           ['Energy / fleet completion time',f'{ga["energy"]:.2f} / {ga["mission_time"]:.2f} s',f'{rs["energy"]:.2f} / {rs["mission_time"]:.2f} s'],
           ['Hazard exposure / valid evaluations',f'{ga["risk"]:.2f} / {run["ga_valid_solutions"]:,}',f'{rs["risk"]:.2f} / {run["random_valid_solutions"]:,}'],
           ['Compute time (seconds)',f'{run["ga_seconds"]:.2f}',f'{run["random_seconds"]:.2f}']],
          [width*.44,width*.28,width*.28])
    heading('Sensitivity and independent repeats')
    seeds = ', '.join(map(str,experiments['seeds']))
    p(f'All 12 sensitivity configurations used {experiments["sensitivity_evaluation_budget"]:,} evaluations each and seeds {seeds} '
      f'({experiments["number_of_sensitivity_runs"]} runs). One factor varies at a time. Equal evaluations, rather than equal generations, '
      'prevent larger populations receiving extra search effort; partial final generations retain previous survivors to keep population size fixed. The table gives mean final fitness. '
      'CSV files also record runtime, diversity, feasible fraction and evaluations to 90% of each run’s observed gain.','SmallBody')
    data = [['Varied parameter','Values → mean final fitness']]
    for family,label in [('population','Population'),('mutation','Mutation'),('tournament','Tournament')]:
        means = experiments['sensitivity'][family]
        data.append([label,'; '.join(f'{float(v):g} → {m:.4f}' for v,m in means.items())])
    table(data,[width*.23,width*.77])
    a,b = experiments['comparison']['GA'],experiments['comparison']['Random Search']
    p(f'Separate repeated comparison ({len(experiments["seeds"])} seeds; {experiments["comparison_evaluation_budget"]:,} evaluations each): '
      f'GA fitness <b>{a["fitness"]["mean"]:.4f} ± {a["fitness"]["std"]:.4f}</b>, random search '
      f'<b>{b["fitness"]["mean"]:.4f} ± {b["fitness"]["std"]:.4f}</b> (sample SD). '
      'These exploratory repeats describe variation across optimizer seeds on one fixed terrain; they do not establish statistical significance or generalization.','SmallBody')
    p('Sensitivity plots include all three factors with error bars, convergence effort, diversity, feasibility and computational cost. '
      'Route-cache reuse differs between optimizers. Wall times were recorded on a shared host with some concurrent jobs; '
      'they are illustrative, not isolated hardware benchmarks. Equal evaluator budgets are not equal CPU budgets.','SmallBody')
    story.append(PageBreak())

    heading('4. Discussion, limitations and conclusion')
    figure('ga_vs_random_search.png',width,218,'Figure 2. Final independently feasible routes and sensing footprints. Red: no-fly; charcoal: buildings; purple contours: hazards; stars: deployments.')
    gain = ga['fitness']-rs['fitness']
    p(f'<b>Evolved strategy.</b> The main GA changes fitness by {ga["fitness"]-run["initial_ga_fitness"]:+.5f} from its initial population and '
      f'finishes {gain:+.5f} relative to equal-budget random search. Weighted coverage is '
      f'{ga["coverage"]:.1%} versus {rs["coverage"]:.1%}; detection is {ga["detection"]:.1%} versus {rs["detection"]:.1%}. '
      'The solution combines deployment and path placement with sensor settings; its performance is measured from expanded routes rather than hardcoded waypoints. '
      f'The sampled-target expected-detection diagnostic is {ga["expected_detections"]:.2f} of {env["target_count"]} targets. '
      'This is an expectation, not an integer count of realized Bernoulli detections.')
    p(f'<b>Trade-offs.</b> Fleet energy totals {ga["energy"]:.2f} model units, completion time is the slowest vehicle’s '
      f'{ga["mission_time"]:.2f} s, and integrated hazard exposure is {ga["risk"]:.2f} hazard-seconds. '
      'Broad active-sensor footprints improve unique coverage, while overlapping observations improve detection but earn no extra coverage. '
      'Faster travel reduces time and exposure, at a propulsion and sensing-dwell cost. A* detours avoid severe hazards when their edge-cost benefit justifies extra distance.')
    best_families = []
    for family,values in experiments['sensitivity'].items():
        key = max(values,key=values.get)
        best_families.append(f'{family}={float(key):g} (mean F={values[key]:.4f})')
    p('<b>Hyperparameters.</b> Best observed means in the individual sweeps were '+ '; '.join(best_families)+'. '
      'These are separate one-factor experiments, not a jointly optimized setting. Population comparisons trade exploration against number of generations under the same budget. '
      'The convergence statistic is relative to each run’s final gain, so a weaker solution can appear to converge sooner. Three seeds give limited precision.')
    p('<b>Strengths and practical limits.</b> The implementation provides deterministic tests, exact accounting of fitness calls, explicit feasibility and auditable JSON/CSV/NPZ artifacts. '
      'The world is static during a mission; footprint sensing assumes no building occlusion and independent exposures. UAV collision avoidance, weather evolution, communication, '
      'acceleration, altitude, turning radius and return-to-base are outside this model. Resource repair can create stationary suffixes and bias search toward feasible prefixes. '
      'All repeated experiments reuse one environment; runtime depends on hardware and route-cache reuse.')
    p('<b>Future work and conclusion.</b> Evaluate several terrain seeds, calibrate energy and sensor models, add time-dependent hazards, reserve energy for return, '
      'and enforce inter-UAV separation. A Pareto-based evolutionary method could expose policy trade-offs instead of fixing weights. '
      'Within the declared surrogate model, the delivered engine executes a real evolutionary search, compares it with an equal-budget baseline and produces feasible, reproducible mission artifacts.')
    p('<b>Reproducibility:</b> python main.py; python main.py --experiments; python generate_report.py; python -m pytest -v. '
      f'Executed with Python {run["python"]}, NumPy {run["numpy"]}. All figures and tables in this report are generated from saved measurements.','SmallBody')

    destination.parent.mkdir(parents=True,exist_ok=True)
    def decorate(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#bed0dc'))
        canvas.line(44,804,A4[0]-44,804)
        canvas.setFont(regular,7)
        canvas.setFillColor(colors.HexColor('#52677b'))
        canvas.drawString(44,814,'ARTIFICIAL INTELLIGENCE | UAV STRATEGY ENGINE')
        canvas.drawString(44,27,'Reproducible simulation · hand-written genetic algorithm')
        canvas.drawRightString(A4[0]-44,27,f'{doc.page}')
        canvas.restoreState()
    doc = SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=48,bottomMargin=44,
                            title='Multi-Objective UAV Strategy Engine Using Genetic Algorithms',author='Team of Three')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(f'Generated {destination.resolve()}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs',help='Directory containing measured results/figures')
    parser.add_argument('--destination',type=Path,default=ROOT/'report'/'Technical_Report.pdf')
    args = parser.parse_args()
    generate(args.output,args.destination)
