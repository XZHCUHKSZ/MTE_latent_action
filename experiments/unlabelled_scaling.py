"""X2. Hold labels fixed and vary only unlabelled observation count."""
from .limited_labels import train_history
from mte.quality import evaluate as endpoint_quality

GRID={'mpe':{'budget':32,'sizes':[32,175,350,700]},
      'mamujoco':{'budget':8,'sizes':[8,24,45,90]}}

def check_design(env,n,budget):
    spec=GRID[env]
    if n not in spec['sizes'] or budget!=spec['budget']:
        raise ValueError('Not a final-paper unlabelled-scaling cell')
    return spec

# Reuse train_history with the explicit N-specific final endpoint artifact.
# Endpoint quality is measured on train-varying coordinates; constant coordinates
# are checked separately. Full-N results are reused, not counted as new seeds.
