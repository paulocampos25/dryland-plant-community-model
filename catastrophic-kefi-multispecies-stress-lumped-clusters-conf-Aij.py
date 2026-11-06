import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#from mpl_toolkits.mplot3d import Axes3D
#from numba import jit

from numpy import loadtxt
from sys import argv


def network(params):
    """Build the 4-neighbor periodic lattice."""
    for i in range(N):
        Node[i, 0] = (N - L + i) if (i < L) else (i - L)              # up
        Node[i, 1] = (i + L) if (i < (N - L)) else (i % L)            # down
        Node[i, 2] = (i + L - 1) if ((i % L) == 0) else (i - 1)       # left
        Node[i, 3] = (i - L + 1) if ((i % L) == (L - 1)) else (i + 1) # right

    return Node

def build_pichon_trait_competition_matrix(Phi, alpha0=1.0, alphae=0.2):
    """
    Build a Pichon-like trait-dependent competition matrix A_ij.

    Phi[i] is the trait/strategy of species i. In this convention,
    larger Phi means more stress-tolerant/facilitating and also more
    sensitive to interspecific competition.

    The matrix A is used together with the b-c trade-off parameter c_i as

        C_i = c_i * sum_j A_ij * rho_j.

    Therefore, the effective competition matrix is

        alpha_eff_ij = c_i * A_ij.

    The Pichon-like structure is

        A_ij = alpha0,                                      if i == j
        A_ij = alphae * (1 + Phi_i * exp(-|Phi_i-Phi_j|)),  if i != j

    where alpha0 controls intraspecific competition and alphae controls
    the basal strength of interspecific competition.
    """
    Phi = np.asarray(Phi, dtype=float)
    N_spec = Phi.size
    A = np.zeros((N_spec, N_spec), dtype=float)

    for i in range(N_spec):
        for j in range(N_spec):
            if i == j:
                A[i, j] = alpha0
            else:
                A[i, j] = alphae

    return A


def count_lumped_species_clusters(State, Node, C, N_spec):
    """
    Count all same-species plant clusters at the final time.

    A cluster is a connected component of sites occupied by the same species.
    The output is lumped over species: it does not return one value per species.

    Returns
    -------
    total_clusters : int
        Total number of same-species clusters, summed over all species.
    max_cluster_size : int
        Size of the largest same-species cluster in the lattice.
    max_cluster_species : int
        Species identity of the largest same-species cluster. Returns -1 if no plant survives.
    """
    N_sites = State.size
    visited = np.zeros(N_sites, dtype=bool)

    total_clusters = 0
    max_cluster_size = 0
    max_cluster_species = -1

    for start in range(N_sites):
        sp = State[start]

        # Ignore fertile/degraded sites and already visited plant sites.
        if sp >= N_spec or visited[start]:
            continue

        total_clusters += 1
        stack = [start]
        visited[start] = True
        cluster_size = 0

        # Depth-first search over sites occupied by the same species.
        while stack:
            site = stack.pop()
            cluster_size += 1

            for jj in range(C[site]):
                nb = Node[site, jj]
                if (not visited[nb]) and State[nb] == sp:
                    visited[nb] = True
                    stack.append(nb)

        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
            max_cluster_species = int(sp)

    return total_clusters, max_cluster_size, max_cluster_species


def dynamics(State, State_new, W, Node, C, params):

    m      = params['m']
    d      = params['d']
    r      = params['r']
    c      = params['c']
    A      = params['A']
    f      = params['f']
    delta  = params['delta']
    b_spec = params['b_species']
    N_spec = params['N_spec']
    Tau    = params['Tau']
    N      = params['N']
    L      = params['L']

    cont_plus = np.bincount(State, minlength=N_spec)[:N_spec]

    # Global density of each species: rho_plus[i] = number of sites occupied by species i / N.
    rho_plus = cont_plus / N

    # Species-specific competition term:
    # C_i = c_i * sum_j A_ij * rho_j.
    # Here, c_i still comes from the b-c trade-off, while A_ij
    # depends on the species traits following the Pichon-like rule.
    competition = c * (A @ rho_plus)

    for i in range(N):

        cont_plus_new = np.zeros(N_spec, dtype=int)
        for j in range(C[i]):
            s = State[Node[i, j]]
            if s < N_spec:
                cont_plus_new[s] += 1

        # Local density of each species in the neighborhood of site i.
        q_plus = cont_plus_new / C[i]

        if State[i] == N_spec:  # fertile site
            # Recruitment uses the trait-dependent competition term:
            # growth_term[i] = max(b_i - c_i * sum_j A_ij * rho_j, 0).
            growth_term = np.maximum(b_spec - competition, 0.0)
            W[i, :N_spec] = (delta * rho_plus + (1 - delta) * q_plus) * growth_term
            W[i, N_spec] = 0
            W[i, N_spec + 1] = d

        if State[i] < N_spec:  # occupied by plant species
            W[i, :N_spec] = 0
            W[i, N_spec] = m
            W[i, N_spec + 1] = 0

        if State[i] == (N_spec + 1):  # degraded site
            sum_f = np.dot(f, q_plus)
            W[i, :N_spec] = 0
            W[i, N_spec] = r + sum_f
            W[i, N_spec + 1] = 0


def Gillespie(State, State_new, W, params):

    N_spec = params['N_spec']
    Tau    = params['Tau']
    N      = params['N']

    sum_rates = W.sum(axis=0)
    n_events = np.random.poisson(sum_rates * Tau)

    counts = np.zeros((N, N_spec + 2), dtype=int)
    State_new = State.copy()

    for k in range(N_spec + 2):
        n_k = n_events[k]
        if n_k > 0 and sum_rates[k] > 0:
            p = W[:, k].astype(float) / sum_rates[k]

            # Remove invalid values and negative probabilities.
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            p = np.maximum(p, 0.0)

            # Renormalize safely.
            psum = p.sum()
            if psum > 0:
                p /= psum

                # Numerical safeguard against p.sum() being slightly greater than 1.
                p[-1] = max(0.0, 1.0 - p[:-1].sum())

                psum = p.sum()
                if psum > 0:
                    p /= psum
                    counts[:, k] = np.random.multinomial(n_k, p)

    for i in range(N):
        ks = np.nonzero(counts[i, :])[0]

        if ks.size == 0:
            continue

        if ks.size == 1:
            State_new[i] = ks[0]
        else:
            local_rates = W[i, ks].astype(float)
            total_rate = local_rates.sum()

            if total_rate > 0:
                probs = local_rates / total_rate
            else:
                local_counts = counts[i, ks].astype(float)
                probs = local_counts / local_counts.sum()

            chosen_k = np.random.choice(ks, p=probs)
            State_new[i] = chosen_k

    return State_new


def function_random_exponential(x):
    r1 = []
    for i in range(x):
        r1.append(np.random.exponential(scale=2, size=None))
    return r1


def function_random_int(x):
    r1 = []
    for i in range(x):
        r1.append(np.random.randint(5, size=None))
    return r1


def function_random_unif(x):
    r1 = []
    for i in range(x):
        r1.append(np.random.random_sample())
    return r1


def function_random_gaussian(x):
    mu = 0
    sigma = 0.1
    r1 = []
    for i in range(x):
        r1.append(np.random.normal(mu, sigma, size=None))
    return r1


if len(argv) != 7:
    print('error: wrong number of arguments')
    print('usage: python script.py L config tmax N_spec S sigma')
    exit()


L = int(argv[1])
config = int(argv[2])
tmax = int(argv[3])
N_spec = int(argv[4])
S = float(argv[5])
sigma = float(argv[6])

m = 0.15
d = 0.025
r = 0.01
delta = 0.1
e = 0.5
f_0 = 0.9
N = L * L
Tau = 0.5
beta = 1.0

# Parameters of the Pichon-like trait-dependent competition matrix A_ij.
# alpha0 controls intraspecific competition; alphae controls basal interspecific competition.
alpha0 = 0.3
alphae = 0.2

C = np.full(N, 4, dtype=int)
W = np.zeros((N, (N_spec + 2)), dtype=float)
State = np.random.randint(0, N_spec, size=N, dtype=int)
State_new = State.copy()
Node = np.zeros((N, 4), dtype=int)

params = dict(
    m=m, d=d, r=r, delta=delta,
    N_spec=N_spec, Tau=Tau, N=N, L=L
)

# Build lattice once; it does not change among configurations.
network(params)

# Output files. No time-series averages and no per-species tables.
# The per-configuration file is opened here and updated at the end of every configuration.
output_lumped = 'LumpedFinalClusters-L%i-S%.2f-sigma%.2f-config%i.txt' % (L, S, sigma, config)
output_summary = 'LumpedFinalSummary-L%i-S%.2f-sigma%.2f-config%i.txt' % (L, S, sigma, config)

with open(output_lumped, 'w') as f_out:
    f_out.write('# Final-time quantities for each configuration. Species are lumped together for density.\n')
    f_out.write('# Clusters are same-species connected components, summed over all species.\n')
    f_out.write('# max_cluster_species is the species identity of the largest same-species cluster.\n')
    f_out.write('# If the system is extinct, max_cluster_species = -1 and b/c/Phi are nan.\n')
    f_out.write('# cfg survivors final_density_all_species total_clusters max_cluster_size max_cluster_species b_largest_cluster_species c_largest_cluster_species Phi_largest_cluster_species\n')
    f_out.flush()

survivors_all_configs = np.zeros(config, dtype=int)
final_density_all_species = np.zeros(config, dtype=float)
total_clusters_all_configs = np.zeros(config, dtype=int)
max_cluster_size_all_configs = np.zeros(config, dtype=int)
max_cluster_species_all_configs = np.full(config, -1, dtype=int)
b_largest_cluster_species_all_configs = np.full(config, np.nan, dtype=float)
c_largest_cluster_species_all_configs = np.full(config, np.nan, dtype=float)
phi_largest_cluster_species_all_configs = np.full(config, np.nan, dtype=float)

print('Data for each configuration:')
print('cfg survivors final_density_all_species total_clusters max_cluster_size max_cluster_species b_largest_cluster_species c_largest_cluster_species Phi_largest_cluster_species')

for cfg in range(config):

    # Independent initial conditions for each configuration.
    State = np.random.randint(0, N_spec, size=N, dtype=int)
    State_new = State.copy()
    W = np.zeros((N, N_spec + 2), dtype=float)

    # Independent random species parameters for each configuration.
    Phi = np.random.uniform(0.0, 1.0, size=N_spec)
    S_stress = S * (1 - e * Phi)
    b_species = beta * (1 - S_stress)
    f = f_0 * Phi
    c = 1.0 - (1.0 - b_species**(1 / sigma))**sigma

    # Trait-dependent competition matrix. The actual effective competition is
    # alpha_eff_ij = c_i * A_ij, because c_i preserves the b-c trade-off.
    A = build_pichon_trait_competition_matrix(Phi, alpha0=alpha0, alphae=alphae)

    params['b_species'] = b_species
    params['f'] = f
    params['c'] = c
    params['A'] = A

    # Save species parameters and competition matrices for this configuration.
  #  species_parameters_file = 'SpeciesParameters-L%i-S%.2f-sigma%.2f-cfg%i.txt' % (L, S, sigma, cfg)
   # Aij_file = 'Aij-PichonLike-L%i-S%.2f-sigma%.2f-cfg%i.txt' % (L, S, sigma, cfg)
  #  alpha_eff_file = 'AlphaEffective-L%i-S%.2f-sigma%.2f-cfg%i.txt' % (L, S, sigma, cfg)

  #  np.savetxt(
   #     species_parameters_file,
  #      np.column_stack((np.arange(N_spec), Phi, b_species, c, f)),
   #     header='species Phi b_i c_i f_i',
  #      fmt=['%d', '%.10f', '%.10f', '%.10f', '%.10f']
   # )
   # np.savetxt(Aij_file, A, header='A_ij Pichon-like trait-dependent competition matrix')
  #  np.savetxt(alpha_eff_file, c[:, None] * A, header='effective alpha_ij = c_i * A_ij')

    for time in range(tmax):

        dynamics(State, State_new, W, Node, C, params)
        State_new = Gillespie(State, State_new, W, params)
        State = State_new.copy()

        cont = np.bincount(State, minlength=N_spec + 2)[:N_spec + 2]

        if cont[:N_spec].sum() == 0:
            break

    # Final state measurements, lumping all species together for density.
    cont_final = np.bincount(State, minlength=N_spec + 2)[:N_spec + 2]
    survivors_all_configs[cfg] = np.sum(cont_final[:N_spec] > 0)
    final_density_all_species[cfg] = cont_final[:N_spec].sum() / N

    # Final spatial clusters: same-species clusters, summed over all species.
    total_clusters, max_cluster_size, max_cluster_species = count_lumped_species_clusters(
        State, Node, C, N_spec
    )

    total_clusters_all_configs[cfg] = total_clusters
    max_cluster_size_all_configs[cfg] = max_cluster_size
    max_cluster_species_all_configs[cfg] = max_cluster_species

    # b and c of the species that forms the largest cluster.
    if max_cluster_species >= 0:
        b_largest_cluster_species_all_configs[cfg] = b_species[max_cluster_species]
        c_largest_cluster_species_all_configs[cfg] = c[max_cluster_species]
        phi_largest_cluster_species_all_configs[cfg] = Phi[max_cluster_species]

    # Print this configuration immediately, before starting the next one.
    print(
        f'{cfg:d} '
        f'{survivors_all_configs[cfg]:d} '
        f'{final_density_all_species[cfg]:.10f} '
        f'{total_clusters_all_configs[cfg]:d} '
        f'{max_cluster_size_all_configs[cfg]:d} '
        f'{max_cluster_species_all_configs[cfg]:d} '
        f'{b_largest_cluster_species_all_configs[cfg]:.10f} '
        f'{c_largest_cluster_species_all_configs[cfg]:.10f} '
        f'{phi_largest_cluster_species_all_configs[cfg]:.10f}',
        flush=True
    )

    # Save this configuration immediately, before starting the next one.
    # This way the output file is updated during the run, not only after all configurations finish.
    with open(output_lumped, 'a') as f_out:
        f_out.write(
            f'{cfg:d} '
            f'{survivors_all_configs[cfg]:d} '
            f'{final_density_all_species[cfg]:.10f} '
            f'{total_clusters_all_configs[cfg]:d} '
            f'{max_cluster_size_all_configs[cfg]:d} '
            f'{max_cluster_species_all_configs[cfg]:d} '
            f'{b_largest_cluster_species_all_configs[cfg]:.10f} '
            f'{c_largest_cluster_species_all_configs[cfg]:.10f} '
            f'{phi_largest_cluster_species_all_configs[cfg]:.10f}\n'
        )
        f_out.flush()

# Summary statistics across configurations.
mean_survivors = survivors_all_configs.mean()
std_survivors = survivors_all_configs.std()
mean_final_density_all_species = final_density_all_species.mean()
std_final_density_all_species = final_density_all_species.std()
mean_total_clusters = total_clusters_all_configs.mean()
std_total_clusters = total_clusters_all_configs.std()
mean_max_cluster_size = max_cluster_size_all_configs.mean()
std_max_cluster_size = max_cluster_size_all_configs.std()

present_largest = max_cluster_species_all_configs >= 0
if present_largest.any():
    mean_b_largest_cluster_species = np.nanmean(b_largest_cluster_species_all_configs)
    std_b_largest_cluster_species = np.nanstd(b_largest_cluster_species_all_configs)
    mean_c_largest_cluster_species = np.nanmean(c_largest_cluster_species_all_configs)
    std_c_largest_cluster_species = np.nanstd(c_largest_cluster_species_all_configs)
    mean_phi_largest_cluster_species = np.nanmean(phi_largest_cluster_species_all_configs)
    std_phi_largest_cluster_species = np.nanstd(phi_largest_cluster_species_all_configs)
else:
    mean_b_largest_cluster_species = np.nan
    std_b_largest_cluster_species = np.nan
    mean_c_largest_cluster_species = np.nan
    std_c_largest_cluster_species = np.nan
    mean_phi_largest_cluster_species = np.nan
    std_phi_largest_cluster_species = np.nan

# Save summary averages in a separate file.
with open(output_summary, 'w') as f_sum:
    f_sum.write('# Summary averages across configurations. Values are calculated from the final state of each configuration. No time averaging is performed.\n')
    f_sum.write('# mean_survivors std_survivors mean_final_density_all_species std_final_density_all_species mean_total_clusters std_total_clusters mean_max_cluster_size std_max_cluster_size mean_b_largest_cluster_species std_b_largest_cluster_species mean_c_largest_cluster_species std_c_largest_cluster_species mean_Phi_largest_cluster_species std_Phi_largest_cluster_species\n')
    f_sum.write(
        f'{mean_survivors:.10f} '
        f'{std_survivors:.10f} '
        f'{mean_final_density_all_species:.10f} '
        f'{std_final_density_all_species:.10f} '
        f'{mean_total_clusters:.10f} '
        f'{std_total_clusters:.10f} '
        f'{mean_max_cluster_size:.10f} '
        f'{std_max_cluster_size:.10f} '
        f'{mean_b_largest_cluster_species:.10f} '
        f'{std_b_largest_cluster_species:.10f} '
        f'{mean_c_largest_cluster_species:.10f} '
        f'{std_c_largest_cluster_species:.10f} '
        f'{mean_phi_largest_cluster_species:.10f} '
        f'{std_phi_largest_cluster_species:.10f}\n'
    )

print('Summary across configurations:')
print('mean_survivors std_survivors mean_final_density_all_species std_final_density_all_species mean_total_clusters std_total_clusters mean_max_cluster_size std_max_cluster_size mean_b_largest_cluster_species std_b_largest_cluster_species mean_c_largest_cluster_species std_c_largest_cluster_species mean_Phi_largest_cluster_species std_Phi_largest_cluster_species')
print(
    f'{mean_survivors:.10f} '
    f'{std_survivors:.10f} '
    f'{mean_final_density_all_species:.10f} '
    f'{std_final_density_all_species:.10f} '
    f'{mean_total_clusters:.10f} '
    f'{std_total_clusters:.10f} '
    f'{mean_max_cluster_size:.10f} '
    f'{std_max_cluster_size:.10f} '
    f'{mean_b_largest_cluster_species:.10f} '
    f'{std_b_largest_cluster_species:.10f} '
    f'{mean_c_largest_cluster_species:.10f} '
    f'{std_c_largest_cluster_species:.10f} '
    f'{mean_phi_largest_cluster_species:.10f} '
    f'{std_phi_largest_cluster_species:.10f}'
)
print('Arquivos salvos:')
print(output_lumped)
print(output_summary)
