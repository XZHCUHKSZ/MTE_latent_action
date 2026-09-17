"""Read-only final-paper index and result inspector; never launches training."""
import argparse,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--list',action='store_true');ap.add_argument('--report',choices=['numeric','visual','route']);ap.add_argument('--check-results',action='store_true')
 a=ap.parse_args()
 if a.list:
  for p in sorted((ROOT/'experiments').glob('*.py')):
   if p.name!='__init__.py':print(p.relative_to(ROOT))
 if a.report=='numeric':
  for r in read('results/numeric/limited_labels.json'):
   full=700 if r['environment']=='mpe' else 90
   if r['n']==full:print(f"{r['environment']:9} B{r['budget']:<3} {r['method']:26} {r['mean']:10.5f} +/- {r['seed_sd']:.5f}")
 if a.report=='visual':
  for r in read('results/visual/summary.json')['rows']:print(f"{r['arm']:45} {r['mean']:9.3f} +/- {r['sd_across_training_seeds']:.3f}")
 if a.report=='route':
  for r in read('results/route/common_comparison.json')['rows']:print(r['environment'],r['name'],round(r['mean_return'],5))
  r=read('results/route/paired_mif_all5.json')['rows'];print('Ma MIF minus Global-Joint16 (all budgets, nested repeats):',statistics.mean(x['mif']-x['global_aux'] for x in r))
 if a.check_results:
  rows=read('provenance/result_lineage.json');err=max(abs(r['paper_value']-r['raw_value']) for r in rows)
  assert err<1e-7
  for r in read('results/numeric/limited_labels.json'):assert abs(statistics.mean(r['seed_returns'])-r['mean'])<1e-7
  print(json.dumps({'references_checked':len(rows),'unique_source_files':len({r['source'] for r in rows}),'max_abs_error':err}))
if __name__=='__main__':main()
