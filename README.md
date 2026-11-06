# Dryland plant community model

Stochastic multispecies vegetation simulation on a periodic square lattice with fertile and degraded sites, competition, and local facilitation. This version fixes **e = 0.5** in the source. The original script is included unchanged.

## Setup and run

Requires Python 3, NumPy and Matplotlib (imported even though no plots are generated).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 catastrophic-kefi-multispecies-stress-lumped-clusters-conf-Aij.py 10 2 20 3 0.5 1.0
```

The six arguments are `L config tmax N_spec S sigma`:

| Argument | Meaning |
| --- | --- |
| L | Lattice side length; L² sites |
| config | Number of independent configurations |
| tmax | Maximum update steps per configuration |
| N_spec | Initial species pool size |
| S | Environmental stress |
| sigma | Recruitment–competition trade-off shape |

The example is a small execution check, not a production experiment. Use positive integer counts, L ≥ 3 for four distinct neighbors, sigma > 0, and parameters keeping b_i in [0,1]; S in [0,1] suffices here.

## Model

Each configuration starts fully occupied by randomly assigned species, with independent traits Phi sampled uniformly from [0,1]. Species identities start at zero.

- Stress: S_i = S × (1 - e × Phi_i).
- Recruitment: b_i = 1 - S_i.
- Facilitation: f_i = 0.9 × Phi_i.
- Competition trade-off: c_i = 1 - (1 - b_i**(1/sigma))**sigma.
- Competition pressure: c_i × sum_j(A_ij × rho_j).

Although comments describe a trait-dependent Pichon-like matrix, the actual implementation uses constant diagonal A_ii = 0.3 and off-diagonal A_ij = 0.2. The matrix itself does not depend on trait differences. Species-specific c_i still scales competition.

Other fixed parameters: m = 0.15, d = 0.025, r = 0.01, delta = 0.1, beta = 1, Tau = 0.5. Each step advances 0.5 model time units; extinction stops a configuration early. The function called Gillespie uses fixed-step Poisson sampling and resolves competing proposed destinations per site.

## Results

Two text files are written in the current working directory:

- `LumpedFinalClusters-L…-S…-sigma…-config….txt`: final-state measurements for each configuration, saved as configurations finish.
- `LumpedFinalSummary-L…-S…-sigma…-config….txt`: means and population standard deviations across configurations.

Per-configuration columns: configuration index, surviving species count, total vegetation density, total same-species cluster count, largest same-species cluster size, its species identity, and that species’ b, c, and Phi. Clusters are connected components using periodic four-neighbor adjacency, counted separately by species and summed. Extinction gives species identity -1 and NaN traits; summary trait statistics exclude extinct configurations. Headers list all columns. No time averaging is performed.

Existing results with matching filenames are overwritten. S and sigma are rounded to two decimal places in filenames; filenames omit e, N_spec, and tmax. Use separate directories for different experiments.

## Reproducibility

No random seed is set or accepted through the command line. Repeated runs differ. Record the source revision, fixed parameters including e, command arguments, and dependency versions. Species-parameter and matrix exports are commented out.
