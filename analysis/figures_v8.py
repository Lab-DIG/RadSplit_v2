"""Generate the v8 balance figure from frozen aggregate results, without raw data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
summary = json.loads((ROOT / 'results/summary.json').read_text())
cohorts = ['covid', 'lidc', 'cbisddsm', 'nlst', 'nih', 'padchest']
names = ['COVID-19', 'LIDC-IDRI', 'CBIS-DDSM', 'NLST', 'NIH', 'PadChest']
methods = ['Group random', 'Stratified group', 'Rerandomization', 'RadSplit']
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for j, (method, color, marker) in enumerate(zip(methods, ['#777777','#d47b25','#7043a0','#007b70'], ['o','s','D','^'])):
    x = [i + (j-1.5)*0.16 for i in range(6)]
    med = [summary['benchmark'][c][method]['mean_abs_smd_median'] for c in cohorts]
    lo = [m-summary['benchmark'][c][method]['mean_abs_smd_q1'] for c,m in zip(cohorts,med)]
    hi = [summary['benchmark'][c][method]['mean_abs_smd_q3']-m for c,m in zip(cohorts,med)]
    ax.errorbar(x, med, yerr=[lo,hi], fmt=marker, color=color, capsize=3, markersize=5, label=method)
ax.set_yscale('log')
ax.set_xticks(range(6),names)
ax.set_ylabel('Mean absolute standardized mean difference')
ax.grid(axis='y',alpha=.22)
ax.spines[['top','right']].set_visible(False)
ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.20),frameon=False,fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'Figure 2.png',dpi=400,bbox_inches='tight')
fig.savefig(OUT/'Figure 2.pdf',bbox_inches='tight')
