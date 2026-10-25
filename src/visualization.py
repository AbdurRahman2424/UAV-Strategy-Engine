"""Publication-friendly figures generated from actual optimizer traces."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/uav-strategy-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                     'figure.dpi':120,'savefig.dpi':180,'axes.spines.top':False,
                     'axes.spines.right':False})
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']


def save(fig,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def mission_map(ax,env,chromosome,value,title,legend=False,coverage=False):
    if not value.feasible:
        raise ValueError('Refusing to display an infeasible flight path as a solution.')
    n = env.size
    extent = (-.5,n-.5,-.5,n-.5)
    ax.imshow(env.priority,origin='lower',cmap='YlOrBr',vmin=1,vmax=6,alpha=.6,extent=extent)
    if coverage:
        ax.imshow(np.ma.masked_where(value.detection_map<=0,value.detection_map),origin='lower',
                  cmap='Blues',vmin=0,vmax=1,alpha=.6,extent=extent)
    levels = [.25,.5,.75]
    ax.contour(env.hazard,levels=levels,colors='#aa3377',linewidths=.65,alpha=.75)
    ax.imshow(np.ma.masked_where(~env.obstacles,env.obstacles),origin='lower',
              cmap=ListedColormap(['#36454f']),vmin=0,vmax=1,extent=extent)
    ax.imshow(np.ma.masked_where(~env.no_fly,env.no_fly),origin='lower',
              cmap=ListedColormap(['#c23b3b']),vmin=0,vmax=1,extent=extent)
    y,x = np.nonzero(env.targets)
    ax.scatter(x,y,marker='x',s=15,color='#171717',linewidth=.8,zorder=4)
    for i,(p,path) in enumerate(zip(chromosome.plans,value.paths)):
        color = COLORS[i%len(COLORS)]
        x,y = np.array(path).T
        ax.plot(x,y,color=color,lw=1.6,label=f'UAV {p.uid+1}',zorder=5)
        wx,wy = np.array(p.waypoints).T
        ax.scatter(wx,wy,color=color,s=19,edgecolor='white',linewidth=.4,zorder=6)
        ax.scatter(*p.start,marker='*',s=145,color=color,edgecolor='black',linewidth=.5,zorder=7)
    ax.set(xlim=(-.5,n-.5),ylim=(-.5,n-.5),xlabel='x (grid cells)',ylabel='y (grid cells)',
           title=title,aspect='equal')
    if legend:
        handles,labels = ax.get_legend_handles_labels()
        handles.extend([Line2D([],[],color='#aa3377',label='Hazard contours'),
                        Line2D([],[],marker='s',ls='',color='#c23b3b',label='No-fly zone'),
                        Line2D([],[],marker='s',ls='',color='#36454f',label='Building / obstacle'),
                        Line2D([],[],marker='s',ls='',color='#f0b04e',label='Priority zones (darker = higher)'),
                        Line2D([],[],marker='x',ls='',color='black',label='Simulated target'),
                        Line2D([],[],marker='*',ls='',color='grey',label='Deployment'),
                        Line2D([],[],marker='o',ls='',color='grey',label='Encoded waypoint')])
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8)


def generate_plots(env,ga,random,output):
    output = Path(output)
    fig,ax = plt.subplots(figsize=(8.5,4.1),layout='constrained')
    x = [h['evaluations'] for h in ga.history]
    ax.plot(x,[h['best_fitness'] for h in ga.history],label='GA best',color=COLORS[0],lw=2)
    ax.plot(x,[h['average_fitness'] for h in ga.history],label='GA population average',color=COLORS[2],lw=1.5)
    ax.plot([h['evaluations'] for h in random.history],[h['best_fitness'] for h in random.history],
            label='Random search best so far',color=COLORS[1],ls='--',lw=2)
    ax.set(xlabel='Fitness evaluations (equal final budget)',ylabel='Normalized weighted fitness',
           title='Fitness convergence | identical environment and evaluation budget')
    ax.grid(alpha=.25)
    ax.legend(loc='lower right')
    # Map real generation checkpoints to the shared evaluation axis.
    secondary = ax.secondary_xaxis('top')
    selected = np.linspace(0,len(ga.history)-1,min(6,len(ga.history)),dtype=int)
    secondary.set_xticks([x[i] for i in selected],[str(ga.history[i]['generation']) for i in selected])
    secondary.set_xlabel('GA generation')
    save(fig,output/'fitness_convergence.png')

    fig,ax = plt.subplots(figsize=(8.9,6),layout='constrained')
    mission_map(ax,env,ga.best,ga.evaluation,
                f'Optimized fleet | fitness {ga.evaluation.fitness:.4f}',legend=True)
    save(fig,output/'optimized_flight_paths.png')

    fig,axes = plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    counts = np.ma.masked_where(~env.traversable,ga.evaluation.observation_counts)
    im = axes[0].imshow(counts,origin='lower',cmap='viridis',vmin=0)
    fig.colorbar(im,ax=axes[0],shrink=.8,label='Sensor observations (overlap included)')
    im = axes[1].imshow(np.ma.masked_where(~env.traversable,ga.evaluation.detection_map),
                        origin='lower',cmap='magma',vmin=0,vmax=1)
    fig.colorbar(im,ax=axes[1],shrink=.8,label='Detection probability')
    for ax,title in zip(axes,['Coverage intensity','Accumulated detection probability']):
        ax.set(title=title,xlabel='x (cells)',ylabel='y (cells)')
        ax.set_facecolor('#8d9498')
    fig.suptitle(f'Fleet sensing | unique weighted coverage {ga.evaluation.coverage:.1%}')
    save(fig,output/'coverage_heatmap.png')

    snapshots = sorted(ga.snapshots.items())
    cols = min(3,len(snapshots))
    rows = int(np.ceil(len(snapshots)/cols))
    fig,axes = plt.subplots(rows,cols,figsize=(cols*4,rows*4),squeeze=False,layout='constrained')
    for ax,(generation,(chromosome,value)) in zip(axes.ravel(),snapshots):
        mission_map(ax,env,chromosome,value,f'Generation {generation}\nF={value.fitness:.3f}; coverage={value.coverage:.1%}',coverage=True)
    for ax in axes.ravel()[len(snapshots):]:
        ax.set_visible(False)
    fig.suptitle('Evolution of the best strategy | blue shading indicates sensing')
    save(fig,output/'evolution_across_generations.png')

    fig,axes = plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    for ax,result,name in zip(axes,[ga,random],['Genetic Algorithm','Random Search']):
        mission_map(ax,env,result.best,result.evaluation,
                    f'{name} | F={result.evaluation.fitness:.4f}\nWeighted coverage {result.evaluation.coverage:.1%}',coverage=True)
    fig.suptitle(f'Equal-budget comparison | {ga.evaluations:,} fitness evaluations each')
    save(fig,output/'ga_vs_random_search.png')
    output_dashboard(output.parent)


def output_dashboard(output):
    """A saved output preview suitable for the README and submission screenshots."""
    import json
    output = Path(output)
    run = json.loads((output/'results'/'run_summary.json').read_text())
    ga,random = run['ga'],run['random']
    fig = plt.figure(figsize=(12,8),layout='constrained')
    grid = fig.add_gridspec(2,2,height_ratios=[1,1.7])
    ax = fig.add_subplot(grid[0,:])
    ax.imshow(plt.imread(output/'figures'/'fitness_convergence.png'))
    ax.axis('off')
    for column,name in enumerate(['optimized_flight_paths.png','coverage_heatmap.png']):
        ax = fig.add_subplot(grid[1,column])
        ax.imshow(plt.imread(output/'figures'/name))
        ax.axis('off')
    fig.suptitle(f'UAV Strategy Engine | GA {ga["fitness"]:.4f} vs random {random["fitness"]:.4f}\n'
                 f'Weighted coverage {ga["coverage"]:.1%} | Detection {ga["detection"]:.1%} | '
                 f'{run["ga_evaluations"]:,} evaluations each',fontsize=15)
    save(fig,output/'screenshots'/'mission_dashboard.png')


def sensitivity_plot(rows,output):
    families = [('population','Population size'),('mutation','Mutation probability'),('tournament','Tournament size')]
    metrics = [('fitness','Final best fitness'),('convergence_evaluations','Evaluations to 90% gain'),
               ('diversity','Final population diversity'),('seconds','Runtime (seconds)'),
               ('feasible_fraction','Feasible evaluation fraction')]
    fig,axes = plt.subplots(len(metrics),3,figsize=(12,14),layout='constrained')
    for col,(family,label) in enumerate(families):
        subset = [r for r in rows if r['parameter']==family]
        xs = sorted(set(float(r['value']) for r in subset))
        for row,(metric,ylabel) in enumerate(metrics):
            means,stds = [],[]
            for x in xs:
                values = [float(r[metric]) for r in subset if float(r['value'])==x]
                means.append(np.mean(values))
                stds.append(np.std(values,ddof=1) if len(values)>1 else 0)
            ax = axes[row,col]
            ax.errorbar(xs,means,yerr=stds,marker='o',capsize=4,color=COLORS[col])
            ax.set(xlabel=label,ylabel=ylabel,xticks=xs)
            ax.grid(alpha=.2)
            if metric=='feasible_fraction':
                ax.set_ylim(.9,1.02)
    fig.suptitle('Hyperparameter sensitivity | equal evaluation budgets; mean ± sample SD across seeds')
    save(fig,Path(output)/'hyperparameter_sensitivity.png')
