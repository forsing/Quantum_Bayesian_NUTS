# Quantum_Bayesian_NUTS 
# NUTS - No-U-Turn Sampler 

 
 
# Quantum_Bayesian_NUTS_V1.py
#
# Optimizovano racunanje ISTOG likelihood-a:
# - samo vazeci kandidati;
# - likelihood i analiticki gradijent u jednom prolazu;
# - najvise cetiri Numba niti;
# - zajednicki kes likelihood-a i gradijenta;
# - NUTS sa adaptacijom pune matrice.
#
# Nema smanjenja istorije ili broja posteriornih uzoraka.
# Nema promena pragova dijagnostike.
#
# Kvantni deo: uslovne amplitude kontrolisanih RY rotacija.
# Sedam uslovnih sest-qubitnih registara i pomocni qubit.
# Lokalna analiticka simulacija, proverena prema Qiskitu.
# NUTS, bootstrap i pretraga su klasicni.
#
# Posteriorna predikcija:
# prosek proizvoda sedam uslovnih verovatnoca.
# NEXT maksimizuje konacnu Monte Carlo aproksimaciju.
#
# Student-t prior odnosi se na parametre, ne na brojeve.
# Bootstrap provere su aproksimativne i ne menjaju NEXT.
#
# Glavni izvori:
# https://jmlr.org/papers/v15/hoffman14a.html
# Westfall / Young, Resampling-Based Multiple Testing, 1993.
# https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
# Haigh, The statistics of the National Lottery, 1997.
# https://doi.org/10.1080/07350015.2000.10524865



from pathlib import Path
from itertools import combinations
from math import comb
from time import perf_counter
import logging
import random
import sys

import numpy as np

random.seed(39)
np.random.seed(39)

try:
    from scipy.special import logsumexp
    from numba import njit, prange, get_num_threads, set_num_threads

    import pytensor
    from pytensor.link.c.cmodule import GCC_compiler
    from pytensor.graph.op import Op

    if sys.platform == "darwin":
        if not getattr(GCC_compiler, "_nuts_v1_linker_fixed", False):
            _original_compile_args = GCC_compiler.compile_args

            def _compatible_compile_args(
                *args, _original=_original_compile_args, **kwargs
            ):
                return [
                    flag
                    for flag in _original(*args, **kwargs)
                    if flag != "-ld64"
                ]

            GCC_compiler.compile_args = staticmethod(
                _compatible_compile_args
            )
            GCC_compiler._nuts_v1_linker_fixed = True

    import pytensor.tensor as pt
    import pymc as pm
    import arviz as az

    az.Numba.disable_numba()

    import faulthandler
    faulthandler.enable(all_threads=True)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, RYGate
    from qiskit.quantum_info import Statevector

except ImportError as exc:
    raise SystemExit(
        "Nedostaje zavisnost: "
        + str(exc)
        + ". Potrebni su NumPy, SciPy, Numba, PyMC 5, "
        + "PyTensor, ArviZ i Qiskit."
    )

if int(pm.__version__.split(".")[0]) != 5:
    raise SystemExit("Ovaj fajl koristi API biblioteke PyMC 5.")

logging.getLogger("pymc").setLevel(logging.ERROR)
set_num_threads(min(4, get_num_threads()))

FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4724_k93.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4724_k93_loto_2984.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4724_k93_loto_plus_1740.csv"),
)

N = 39
K = 7
SEED = 39

PRIOR_SCALES = (0.35, 1.0)
CHAINS = 2
DRAWS = 600
TUNE = 900
POSTERIOR_POINTS_PER_CHAIN = 64

TARGET_ACCEPT = 0.95
MAX_RHAT = 1.05
MIN_ESS = 100.0
MIN_BFMI = 0.20

BOOTSTRAP_REPLICATES = 1999
ALPHA = 0.05
MAX_RUN = 4
RELATIVE_SEARCH_TOLERANCE = 1e-10
DETAILS = False

FEATURE_NAMES = (
    "poslednje kolo",
    "promena vremenske memorije",
    "vreme od pojave",
    "vremenske parne veze",
)


def load_csv(path):
    if not path.is_file():
        raise ValueError("CSV nije pronadjen: " + str(path))

    data = None
    for delimiter in (",", ";", None):
        try:
            candidate = np.loadtxt(
                path, delimiter=delimiter, dtype=np.float64, ndmin=2
            )
            if candidate.shape[1] == K:
                data = candidate
                break
        except ValueError:
            pass

    if data is None:
        raise ValueError("CSV mora imati sedam numerickih kolona bez hedera.")
    if len(data) < 120:
        raise ValueError("Potrebno je najmanje 120 istorijskih kola.")
    if not np.all(np.isfinite(data)):
        raise ValueError("CSV sadrzi neispravne vrednosti.")
    if not np.all(data == np.rint(data)):
        raise ValueError("CSV sadrzi brojeve koji nisu celi.")
    if np.any((data < 1) | (data > N)):
        raise ValueError("Brojevi moraju biti od 1 do 39.")

    data = np.sort(data.astype(np.int64), axis=1)
    if np.any(np.diff(data, axis=1) <= 0):
        raise ValueError("Red CSV-a sadrzi duplikate.")

    return data


def log_completion_row(n, remaining, last):
    result = np.full(n, -np.inf)
    for value in range(last + 1, n - remaining + 2):
        result[value - 1] = np.log(
            float(comb(n - value, remaining - 1))
        )
    return result


def make_dataset(draws):
    count = len(draws) - 1
    features = np.empty((count, K, N, 4))
    bases = np.empty((count, K, N))
    targets = draws[1:] - 1

    last_seen = np.full(N, -1, dtype=np.int64)
    short = np.zeros(N)
    long = np.zeros(N)
    marginal = np.zeros(N)
    joint = np.zeros((N, N))

    for t, draw in enumerate(draws):
        present = np.zeros(N)
        present[draw - 1] = 1.0
        outer = np.outer(present, present)

        if t == 0:
            short = present.copy()
            long = present.copy()
            marginal = present.copy()
            joint = outer.copy()
        else:
            short = 0.80 * short + 0.20 * present
            long = 0.98 * long + 0.02 * present
            marginal = 0.95 * marginal + 0.05 * present
            joint = 0.95 * joint + 0.05 * outer

        last_seen[draw - 1] = t
        known = last_seen >= 0
        waiting = np.zeros(N)
        ages = t - last_seen[known]
        waiting[known] = ages / (ages + 6.0)
        waiting[known] -= waiting[known].mean()

        nodes = np.column_stack((
            present - present.mean(),
            4.0 * (short - long),
            waiting,
        ))
        pairs = 4.0 * (joint - np.outer(marginal, marginal))

        if t < count:
            last = 0
            for position in range(K):
                remaining = K - position
                features[t, position, :, :3] = nodes
                features[t, position, :, 3] = (
                    pairs[last - 1] if last else 0.0
                )
                bases[t, position] = log_completion_row(
                    N, remaining, last
                )
                last = int(draws[t + 1, position])

    return features, bases, targets, nodes.copy(), pairs.copy()


@njit(nogil=True, parallel=True)
def packed_likelihood(theta, x, weights, offsets, chosen, need_gradient):
    """
    Ista amplituda kao ranije:
      a(z) = cos(pi/4 + 0.625*tanh(z))**2
           = (1 - sin(1.25*tanh(z)))/2

    Izvod:
      a'(z) = -0.625*cos(1.25*tanh(z))*(1-tanh(z)**2)

    Uslovni logp:
      log(w_y*a_y) - log(sum_v w_v*a_v)

    Zajednicko skaliranje svih w u jednom uslovu se ponistava.
    """
    groups = len(chosen)
    terms = np.empty(groups)
    blocks = 16
    partial_gradient = np.zeros((blocks, 4))

    for block in prange(blocks):
        start_group = groups * block // blocks
        stop_group = groups * (block + 1) // blocks

        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        g3 = 0.0

        for group in range(start_group, stop_group):
            denominator = 0.0
            d0 = 0.0
            d1 = 0.0
            d2 = 0.0
            d3 = 0.0
            selected_a = 1.0
            selected_da = 0.0
            selected = chosen[group]

            for j in range(offsets[group], offsets[group + 1]):
                z = (
                    theta[0] * x[j, 0]
                    + theta[1] * x[j, 1]
                    + theta[2] * x[j, 2]
                    + theta[3] * x[j, 3]
                )
                h = np.tanh(z)
                angle = 1.25 * h
                a = 0.5 * (1.0 - np.sin(angle))
                denominator += weights[j] * a

                if j == selected:
                    selected_a = a

                if need_gradient:
                    da = -0.625 * np.cos(angle) * (1.0 - h * h)
                    weighted_da = weights[j] * da
                    d0 += weighted_da * x[j, 0]
                    d1 += weighted_da * x[j, 1]
                    d2 += weighted_da * x[j, 2]
                    d3 += weighted_da * x[j, 3]

                    if j == selected:
                        selected_da = da

            terms[group] = (
                np.log(weights[selected])
                + np.log(selected_a)
                - np.log(denominator)
            )

            if need_gradient:
                ratio = selected_da / selected_a
                g0 += ratio * x[selected, 0] - d0 / denominator
                g1 += ratio * x[selected, 1] - d1 / denominator
                g2 += ratio * x[selected, 2] - d2 / denominator
                g3 += ratio * x[selected, 3] - d3 / denominator

        partial_gradient[block, 0] = g0
        partial_gradient[block, 1] = g1
        partial_gradient[block, 2] = g2
        partial_gradient[block, 3] = g3

    gradient = np.zeros(4)
    for block in range(blocks):
        for j in range(4):
            gradient[j] += partial_gradient[block, j]

    return terms, gradient


class PackedLikelihood:
    def __init__(self, features, bases, targets):
        self.rows = len(features)
        self.positions = features.shape[1]

        flat_x = features.reshape(-1, features.shape[2], 4)
        flat_b = bases.reshape(-1, bases.shape[2])
        flat_y = targets.reshape(-1)
        mask = np.isfinite(flat_b)

        counts = mask.sum(axis=1).astype(np.int64)
        if np.any(counts == 0):
            raise ValueError("Prazna podrska uslovne raspodele.")

        group_indices = np.arange(len(flat_y))
        if not np.all(mask[group_indices, flat_y]):
            raise ValueError("Cilj nije u podrsci uslovne raspodele.")

        self.offsets = np.empty(len(counts) + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(counts, out=self.offsets[1:])

        local_indices = (
            np.cumsum(mask, axis=1)[group_indices, flat_y] - 1
        )
        self.chosen = np.ascontiguousarray(
            self.offsets[:-1] + local_indices, dtype=np.int64
        )

        maxima = flat_b.max(axis=1)
        scaled = np.exp(flat_b - maxima[:, None])
        self.weights = np.ascontiguousarray(scaled[mask])
        self.x = np.ascontiguousarray(flat_x[mask])

        self.last_theta = None
        self.last_value = None
        self.last_gradient = None

    def raw(self, theta, need_gradient=True):
        return packed_likelihood(
            np.ascontiguousarray(theta, dtype=np.float64),
            self.x,
            self.weights,
            self.offsets,
            self.chosen,
            need_gradient,
        )

    def evaluate(self, theta):
        theta = np.asarray(theta, dtype=np.float64)

        if not np.all(np.isfinite(theta)):
            return -np.inf, np.zeros(4)

        if self.last_theta is None or not np.array_equal(
            theta, self.last_theta
        ):
            terms, gradient = self.raw(theta, True)
            self.last_theta = theta.copy()
            self.last_value = float(terms.sum())
            self.last_gradient = gradient.copy()

        return self.last_value, self.last_gradient

    def logp_rows(self, theta):
        terms, _ = self.raw(theta, False)
        return terms.reshape(self.rows, self.positions).sum(axis=1)


class LikelihoodGradientOp(Op):
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, backend):
        self.backend = backend

    def perform(self, node, inputs, outputs):
        _, gradient = self.backend.evaluate(inputs[0])
        outputs[0][0] = np.asarray(gradient, dtype=np.float64).copy()


class LikelihoodOp(Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, backend):
        self.backend = backend
        self.gradient_op = LikelihoodGradientOp(backend)

    def perform(self, node, inputs, outputs):
        value, _ = self.backend.evaluate(inputs[0])
        outputs[0][0] = np.asarray(value, dtype=np.float64)

    def grad(self, inputs, output_gradients):
        return [output_gradients[0] * self.gradient_op(inputs[0])]


def reference_logp(theta, features, bases, targets):
    field = np.einsum("...f,f->...", features, theta)
    phase = np.pi / 2.0 + 1.25 * np.tanh(field)
    scores = bases + 2.0 * np.log(np.cos(phase / 2.0))
    selected = np.take_along_axis(
        scores, targets[..., None], axis=2
    )[..., 0]
    return np.sum(selected - logsumexp(scores, axis=2), axis=1)


def predictive_logp(samples, backend):
    accumulator = np.full(backend.rows, -np.inf)
    for theta in samples:
        accumulator = np.logaddexp(
            accumulator, backend.logp_rows(theta)
        )
    return accumulator - np.log(len(samples))


def verify_fast_likelihood(features, bases, targets):
    features = features[:8]
    bases = bases[:8]
    targets = targets[:8]
    backend = PackedLikelihood(features, bases, targets)

    points = (
        np.zeros(4),
        np.array([0.17, -0.23, 0.11, 0.19]),
        np.array([1.1, -1.7, 0.8, -0.9]),
        np.array([8.0, -6.0, 4.0, -7.0]),
    )

    for theta in points:
        value, gradient = backend.evaluate(theta)
        reference_rows = reference_logp(
            theta, features, bases, targets
        )
        if not np.allclose(
            backend.logp_rows(theta),
            reference_rows,
            rtol=1e-11,
            atol=1e-10,
        ):
            raise RuntimeError("Brzi likelihood menja raspodelu.")

        if not np.isclose(
            value, reference_rows.sum(), rtol=1e-11, atol=1e-9
        ):
            raise RuntimeError("Brzi zbir likelihood-a nije ispravan.")

        numerical = np.empty(4)
        for j in range(4):
            step = 1e-5 * max(1.0, abs(theta[j]))
            plus = theta.copy()
            minus = theta.copy()
            plus[j] += step
            minus[j] -= step
            numerical[j] = (
                reference_logp(plus, features, bases, targets).sum()
                - reference_logp(minus, features, bases, targets).sum()
            ) / (2.0 * step)

        error = np.max(
            np.abs(gradient - numerical)
            / np.maximum(1.0, np.abs(numerical))
        )
        if error > 2e-5:
            raise RuntimeError("Analiticki gradijent nije prosao proveru.")

    theta_symbol = pt.dvector("theta_check")
    value_symbol = LikelihoodOp(backend)(theta_symbol)
    function = pytensor.function(
        [theta_symbol],
        [value_symbol, pt.grad(value_symbol, theta_symbol)],
    )
    theta = points[1]
    actual_value, actual_gradient = function(theta)
    expected_value, expected_gradient = backend.evaluate(theta)

    if not np.isclose(actual_value, expected_value, atol=1e-10):
        raise RuntimeError("PyTensor veza likelihood-a nije ispravna.")
    if not np.allclose(
        actual_gradient, expected_gradient, rtol=1e-11, atol=1e-10
    ):
        raise RuntimeError("PyTensor veza gradijenta nije ispravna.")





def posterior_diagnostics(idata):
    rhat_values = np.asarray(
        az.rhat(idata, var_names=["theta"])["theta"].values,
        dtype=np.float64,
    )
    bulk_values = np.asarray(
        az.ess(
            idata,
            var_names=["theta"],
            method="bulk",
        )["theta"].values,
        dtype=np.float64,
    )
    tail_values = np.asarray(
        az.ess(
            idata,
            var_names=["theta"],
            method="tail",
        )["theta"].values,
        dtype=np.float64,
    )
    bfmi_values = np.asarray(
        az.bfmi(idata),
        dtype=np.float64,
    )

    rhat = float(np.max(rhat_values))
    bulk = float(np.min(bulk_values))
    tail = float(np.min(tail_values))
    bfmi = float(np.min(bfmi_values))

    divergences = int(
        idata.sample_stats["diverging"].values.sum()
    )

    finite = all(
        np.all(np.isfinite(values))
        for values in (
            rhat_values,
            bulk_values,
            tail_values,
            bfmi_values,
        )
    )

    passed = (
        finite
        and rhat <= MAX_RHAT
        and bulk >= MIN_ESS
        and tail >= MIN_ESS
        and bfmi >= MIN_BFMI
        and divergences == 0
    )

    return passed, rhat, bulk, tail, bfmi, divergences






def benchmark_backend(backend):
    times = []
    for i in range(4):
        theta = np.array([0.17, -0.23, 0.11, 0.19]) + i * 0.013
        start = perf_counter()
        backend.raw(theta, True)
        elapsed = perf_counter() - start
        if i:
            times.append(elapsed)

    print(
        "Izmereno racunanje likelihood-a i gradijenta: "
        f"{1000.0 * np.median(times):.2f} ms "
        f"| Numba niti: {get_num_threads()}."
    )


def fit_posterior(backend, prior_scale):
    likelihood_op = LikelihoodOp(backend)
    last_report = None

    for attempt in range(2):
        multiplier = 1 if attempt == 0 else 2
        if attempt:
            print("NUTS: dodatno uzorkovanje zbog dijagnostike lanaca.")

        with pm.Model():
            theta = pm.StudentT(
                "theta", nu=4.0, mu=0.0, sigma=prior_scale, shape=4
            )
            pm.Potential(
                "whole_combination_log_likelihood",
                likelihood_op(pt.cast(theta, "float64")),
            )
            idata = pm.sample(
                draws=DRAWS * multiplier,
                tune=TUNE * multiplier,
                chains=CHAINS,
                cores=1,
                random_seed=SEED,
                init="adapt_full",
                initvals={"theta": np.zeros(4, dtype=np.float64)},
                target_accept=TARGET_ACCEPT if attempt == 0 else 0.98,
                nuts_sampler="pymc",
                progressbar=True,
                compute_convergence_checks=False,
                return_inferencedata=True,
            )


        print(
            "Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.",
            flush=True,
        )
        last_report = posterior_diagnostics(idata)
        print("Dijagnostika izracunata.", flush=True)


        if last_report[0]:
            values = np.asarray(
                idata.posterior["theta"].values, dtype=np.float64
            )
            indices = np.linspace(
                0,
                values.shape[1] - 1,
                POSTERIOR_POINTS_PER_CHAIN,
                dtype=np.int64,
            )
            samples = values[:, indices, :].reshape(-1, 4)
            if not np.all(np.isfinite(samples)):
                raise RuntimeError("Neispravni posteriorni uzorci.")
            return np.ascontiguousarray(samples)

    _, rhat, bulk, tail, bfmi, divergences = last_report
    raise RuntimeError(
        "NUTS dijagnostika nije prosla: "
        f"Rhat={rhat:.4f}, ESS={min(bulk, tail):.1f}, "
        f"BFMI={bfmi:.3f}, divergencije={divergences}."
    )


def transition_table(samples, nodes, pairs, n=N, k=K):
    q = np.zeros((len(samples), k + 1, n + 1, n + 1))

    for remaining in range(1, k + 1):
        for last in range(n - remaining + 1):
            values = np.arange(last + 1, n - remaining + 2)
            f = np.empty((len(values), 4))
            f[:, :3] = nodes[values - 1]
            f[:, 3] = (
                pairs[last - 1, values - 1] if last else 0.0
            )

            phase = np.pi / 2.0 + 1.25 * np.tanh(samples @ f.T)
            log_base = np.array([
                np.log(float(comb(n - int(v), remaining - 1)))
                for v in values
            ])
            scores = (
                log_base[None, :]
                + 2.0 * np.log(np.cos(phase / 2.0))
            )
            probabilities = np.exp(
                scores - logsumexp(scores, axis=1)[:, None]
            )
            q[:, remaining, last, values] = probabilities

    return np.ascontiguousarray(q)


@njit
def best_completion_bounds(q, n, k):
    count = q.shape[0]
    bounds = np.zeros((count, k + 1, n + 1))
    bounds[:, 0, :] = 1.0

    for remaining in range(1, k + 1):
        for last in range(n - remaining + 1):
            for s in range(count):
                best = 0.0
                for value in range(last + 1, n - remaining + 2):
                    probability = (
                        q[s, remaining, last, value]
                        * bounds[s, remaining - 1, value]
                    )
                    if probability > best:
                        best = probability
                bounds[s, remaining, last] = best
    return bounds


@njit
def search_branch(
    q, bounds, n, remaining, last, run, depth,
    mass, path, best_path, best_score, visited, max_run, tolerance
):
    count = len(mass)
    upper = 0.0
    for s in range(count):
        upper += mass[s] * bounds[s, remaining, last]
    upper /= count

    if best_score[0] > 0.0:
        if upper <= best_score[0] * (1.0 + tolerance):
            return

    if remaining == 0:
        value = mass.sum() / count
        visited[0] += 1
        if value > best_score[0]:
            best_score[0] = value
            for j in range(len(path)):
                best_path[j] = path[j]
        return

    candidates = np.empty(n, dtype=np.int64)
    priorities = np.empty(n)
    length = 0

    for value in range(last + 1, n - remaining + 2):
        new_run = run + 1 if depth > 0 and value == last + 1 else 1
        if new_run > max_run:
            continue

        bound = 0.0
        for s in range(count):
            bound += (
                mass[s]
                * q[s, remaining, last, value]
                * bounds[s, remaining - 1, value]
            )
        candidates[length] = value
        priorities[length] = bound / count
        length += 1

    order = np.argsort(priorities[:length])
    for index in range(length - 1, -1, -1):
        position = order[index]
        if best_score[0] > 0.0:
            if priorities[position] <= best_score[0] * (1.0 + tolerance):
                continue

        value = candidates[position]
        new_run = run + 1 if depth > 0 and value == last + 1 else 1
        next_mass = np.empty(count)
        for s in range(count):
            next_mass[s] = mass[s] * q[s, remaining, last, value]

        path[depth] = value
        search_branch(
            q, bounds, n, remaining - 1, value, new_run, depth + 1,
            next_mass, path, best_path, best_score, visited,
            max_run, tolerance,
        )


def allowed_run(combination):
    run = 1
    for a, b in zip(combination[:-1], combination[1:]):
        run = run + 1 if b == a + 1 else 1
        if run > MAX_RUN:
            return False
    return True


def decode_next(q, n=N, k=K, tolerance=RELATIVE_SEARCH_TOLERANCE):
    bounds = best_completion_bounds(q, n, k)
    best_path = np.zeros(k, dtype=np.int64)
    best_score = np.array([-1.0])
    visited = np.zeros(1, dtype=np.int64)

    search_branch(
        q, bounds, n, k, 0, 0, 0,
        np.ones(q.shape[0]),
        np.zeros(k, dtype=np.int64),
        best_path, best_score, visited, MAX_RUN, tolerance,
    )

    if (
        best_score[0] <= 0.0
        or len(np.unique(best_path)) != k
        or np.any((best_path < 1) | (best_path > n))
        or not allowed_run(best_path)
    ):
        raise RuntimeError("Neispravan rezultat dekodiranja.")

    return best_path, float(best_score[0]), int(visited[0])


def combination_probability(q, combination):
    mass = np.ones(q.shape[0])
    last = 0
    remaining = len(combination)
    for value in combination:
        mass *= q[:, remaining, last, value]
        remaining -= 1
        last = value
    return float(mass.mean())


def adjusted_fdr(pvalues, dependence=False):
    pvalues = np.asarray(pvalues)
    m = len(pvalues)
    order = np.argsort(pvalues)
    factor = (
        np.sum(1.0 / np.arange(1, m + 1)) if dependence else 1.0
    )
    ranked = pvalues[order] * m * factor / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def block_bootstrap_diagnostics(gains):
    # Uslovna ablacija fiksno naucenog modela.
    # Aproksimativno tumacenje zahteva dovoljno stabilan proces
    # i slabu dugorocnu zavisnost. Nije egzaktna FWER garancija.
    length, family = gains.shape
    block = max(2, int(np.ceil(length ** (1.0 / 3.0))))
    blocks_needed = int(np.ceil(length / block))
    means = gains.mean(axis=0)
    centered = gains - means

    rng = np.random.default_rng(SEED)
    bootstrap_means = np.empty((BOOTSTRAP_REPLICATES, family))
    offsets = np.arange(block)

    for b in range(BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, length, size=blocks_needed)
        indices = (
            (starts[:, None] + offsets) % length
        ).ravel()[:length]
        bootstrap_means[b] = centered[indices].mean(axis=0)

    standard_errors = bootstrap_means.std(axis=0, ddof=1)
    usable = standard_errors > 1e-14
    observed = np.zeros(family)
    simulated = np.zeros_like(bootstrap_means)
    observed[usable] = means[usable] / standard_errors[usable]
    simulated[:, usable] = (
        bootstrap_means[:, usable] / standard_errors[usable]
    )

    raw = np.ones(family)
    maxt = np.ones(family)
    maxima = simulated.max(axis=1)
    for j in range(family):
        if usable[j]:
            raw[j] = (
                1 + np.count_nonzero(simulated[:, j] >= observed[j])
            ) / (BOOTSTRAP_REPLICATES + 1)
            maxt[j] = (
                1 + np.count_nonzero(maxima >= observed[j])
            ) / (BOOTSTRAP_REPLICATES + 1)

    return {
        "mean_gain": means,
        "maxT": maxt,
        "BH": adjusted_fdr(raw),
        "BY": adjusted_fdr(raw, dependence=True),
    }


def verify_qiskit_amplitudes():
    values = np.arange(1, N + 1)
    nodes = np.column_stack((
        np.sin(values * 0.31),
        np.cos(values * 0.17),
        np.sin(values * 0.11),
    ))
    theta = np.array([[0.2, -0.3, 0.15, 0.1]])
    pairs = 0.1 * np.cos(values[:, None] - values[None, :])
    q = transition_table(theta, nodes, pairs)

    # Prvni podmineny registr a jeden registr s nenulovym prefixem.
    for remaining, last in ((K, 0), (4, 9)):
        base = np.zeros(64)
        for value in range(last + 1, N - remaining + 2):
            base[value] = float(comb(N - value, remaining - 1))
        base /= base.sum()

        circuit = QuantumCircuit(7)
        circuit.append(StatePreparation(np.sqrt(base)), list(range(6)))

        for value in range(last + 1, N - remaining + 2):
            f = np.empty(4)
            f[:3] = nodes[value - 1]
            f[3] = pairs[last - 1, value - 1] if last else 0.0
            phase = np.pi / 2.0 + 1.25 * np.tanh(theta[0] @ f)
            circuit.append(
                RYGate(float(phase)).control(6, ctrl_state=value),
                list(range(7)),
            )

        state = Statevector.from_instruction(circuit).data
        probabilities = np.abs(state[:64]) ** 2
        probabilities /= probabilities.sum()

        reference = np.zeros(64)
        reference[:N + 1] = q[0, remaining, last]
        if not np.allclose(
            probabilities, reference, rtol=1e-9, atol=1e-11
        ):
            raise RuntimeError("Qiskit provera amplituda nije prosla.")


def internal_checks():
    theta = np.array([
        [0.21, -0.32, 0.17, 0.29],
        [-0.15, 0.11, 0.33, -0.27],
        [0.08, 0.19, -0.23, 0.12],
    ])

    for n, k in ((8, 3), (8, 5)):
        values = np.arange(1, n + 1, dtype=np.float64)
        nodes = np.column_stack((
            np.sin(values),
            np.cos(values * 0.4),
            np.sin(values * 0.7),
        ))
        pairs = 0.2 * np.cos(values[:, None] - values[None, :])
        q = transition_table(theta, nodes, pairs, n, k)

        total = 0.0
        brute_probability = -1.0
        for combination in combinations(range(1, n + 1), k):
            probability = combination_probability(q, combination)
            total += probability
            if allowed_run(combination):
                brute_probability = max(brute_probability, probability)

        if not np.isclose(total, 1.0, atol=1e-11, rtol=1e-11):
            raise RuntimeError("Raspodela nije normalizovana.")

        _, score, _ = decode_next(q, n, k, tolerance=0.0)
        if not np.isclose(
            score, brute_probability, rtol=1e-11, atol=1e-13
        ):
            raise RuntimeError("Pretraga ne odgovara iscrpnoj proveri.")

        zero_q = transition_table(np.zeros((1, 4)), nodes, pairs, n, k)
        expected = 1.0 / comb(n, k)
        for combination in combinations(range(1, n + 1), k):
            actual = combination_probability(zero_q, combination)
            if not np.isclose(
                actual, expected, rtol=1e-11, atol=1e-13
            ):
                raise RuntimeError("Pristrasnost sortiranih pozicija.")

    synthetic = np.array([
        sorted({((t * 5 + j * 3) % N) + 1 for j in range(K)})
        for t in range(18)
    ], dtype=np.int64)

    full = make_dataset(synthetic)
    prefix = make_dataset(synthetic[:12])
    if not np.allclose(full[0][:11], prefix[0]):
        raise RuntimeError("Buduci podaci menjaju ranije karakteristike.")
    if not np.array_equal(full[1][:11], prefix[1]):
        raise RuntimeError("Buduci podaci menjaju raniju podrsku.")

    verify_fast_likelihood(prefix[0], prefix[1], prefix[2])
    verify_qiskit_amplitudes()


def process_csv(path):
    print("CSV:", path.name)
    draws = load_csv(path)
    features, bases, targets, nodes, pairs = make_dataset(draws)
    total = len(targets)
    train_end = int(0.70 * total)
    validation_end = int(0.85 * total)

    train_backend = PackedLikelihood(
        features[:train_end], bases[:train_end], targets[:train_end]
    )
    validation_backend = PackedLikelihood(
        features[train_end:validation_end],
        bases[train_end:validation_end],
        targets[train_end:validation_end],
    )
    benchmark_backend(train_backend)

    selected_scale = None
    best_validation = np.inf

    for scale in PRIOR_SCALES:
        print(f"Ucenje posteriora: prior={scale:g}.", flush=True)
        try:
            samples = fit_posterior(train_backend, scale)
            validation = -predictive_logp(
                samples, validation_backend
            ).mean()
            if not np.isfinite(validation):
                raise RuntimeError("Validacioni NLL nije konacan.")
            if validation < best_validation:
                best_validation = float(validation)
                selected_scale = scale
        except RuntimeError as exc:
            print(f"PROVERA: prior={scale:g} odbacen: {exc}")

    if selected_scale is None:
        raise RuntimeError("Nijedan posterior nije prosao dijagnostiku.")

    print(
        f"Validacija NLL: {best_validation:.6f} "
        f"| prior: {selected_scale:g}"
    )
    del train_backend, validation_backend

    print("Ucenje modela za test.", flush=True)
    fit_backend = PackedLikelihood(
        features[:validation_end],
        bases[:validation_end],
        targets[:validation_end],
    )
    test_samples = fit_posterior(fit_backend, selected_scale)
    del fit_backend

    test_backend = PackedLikelihood(
        features[validation_end:],
        bases[validation_end:],
        targets[validation_end:],
    )
    full_logp = predictive_logp(test_samples, test_backend)
    print(f"Test NLL: {-full_logp.mean():.6f}")

    gains = np.empty((test_backend.rows, 4))
    for j in range(4):
        ablated = test_samples.copy()
        ablated[:, j] = 0.0
        gains[:, j] = (
            full_logp - predictive_logp(ablated, test_backend)
        )

    diagnostics = block_bootstrap_diagnostics(gains)
    print(
        "Ablaciona provera, priblizni blok-bootstrap: "
        f"maxT={np.count_nonzero(diagnostics['maxT'] <= ALPHA)}/4 "
        f"| BY={np.count_nonzero(diagnostics['BY'] <= ALPHA)}/4."
    )

    if DETAILS:
        for j, name in enumerate(FEATURE_NAMES):
            print(
                f"{name}: doprinos={diagnostics['mean_gain'][j]:.6f} "
                f"| maxT={diagnostics['maxT'][j]:.4f} "
                f"| BH={diagnostics['BH'][j]:.4f} "
                f"| BY={diagnostics['BY'][j]:.4f}"
            )
    del test_backend

    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)
    final_backend = PackedLikelihood(features, bases, targets)
    final_samples = fit_posterior(final_backend, selected_scale)
    del final_backend

    q = transition_table(final_samples, nodes, pairs)
    print(
        "NEXT: deterministicka pretraga posteriorne aproksimacije "
        f"sa {len(final_samples)} uzoraka parametara.",
        flush=True,
    )
    prediction, probability, visited = decode_next(q)

    checked = combination_probability(q, prediction)
    if not np.isclose(
        probability, checked, rtol=1e-11, atol=1e-15
    ):
        raise RuntimeError("Provera NEXT verovatnoce nije prosla.")

    print("NEXT:", prediction.tolist())
    if DETAILS:
        print("Provereni listovi pretrage:", visited)
        print(f"Verovatnoca u aproksimaciji: {probability:.12g}")

    return {
        "name": path.name,
        "draws": draws,
        "next": tuple(int(v) for v in prediction),
        "context": np.concatenate((nodes.ravel(), pairs.ravel())),
        "parameters": final_samples.mean(axis=0),
    }


def main():
    print("Quantum Bayesian NUTS V1: optimizovan likelihood i gradijent.")
    print("Lokalne kvantne amplitude; klasicni NUTS i bootstrap.")
    print("NLL cele sedmorke; posteriorna integracija je aproksimativna.")
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    internal_checks()
    print("Interne provere: prosle.", flush=True)

    results = []
    failures = 0

    for path in FILES:
        print()
        try:
            results.append(process_csv(path))
        except KeyboardInterrupt:
            print()
            print("Obrada prekinuta korisnickim zahtevom.")
            return 130
        except Exception as exc:
            failures += 1
            print("STOP za ovaj CSV:", str(exc))

    for i in range(len(results)):
        for j in range(i):
            first = results[j]
            second = results[i]
            if first["next"] != second["next"]:
                continue

            same_input = (
                first["draws"].shape == second["draws"].shape
                and np.array_equal(first["draws"], second["draws"])
            )
            context_difference = np.linalg.norm(
                first["context"] - second["context"]
            )
            parameter_difference = np.linalg.norm(
                first["parameters"] - second["parameters"]
            )
            print()
            print(
                "PROVERA: isti NEXT za "
                + first["name"] + " i " + second["name"] + "."
            )
            print(
                f"Isti CSV sadrzaj: {same_input} "
                f"| razlika konteksta: {context_difference:.6g} "
                f"| razlika posteriornih proseka: {parameter_difference:.6g}."
            )

    if failures:
        print()
        print("Obrada nije kompletna: neki CSV-ovi nisu prosli provere.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")
        sys.exit(130)
    except Exception as exc:
        print("STOP:", str(exc))
        sys.exit(1)



"""
Quantum Bayesian NUTS V1: optimizovan likelihood i gradijent.
Lokalne kvantne amplitude; klasicni NUTS i bootstrap.
NLL cele sedmorke; posteriorna integracija je aproksimativna.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.



CSV: loto7_4724_k93.csv
Izmereno racunanje likelihood-a i gradijenta: 2.10 ms | Numba niti: 4.
Ucenje posteriora: prior=0.35.
                                                                                                            
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.562       7            49.05 draws/s   0:00:30   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.630       7            25.85 draws/s   0:00:58   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Ucenje posteriora: prior=1.
                                                                                                         
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.492       7            48.69 draws/s   0:00:30   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.694       7            25.55 draws/s   0:00:58   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Validacija NLL: 16.548854 | prior: 1
Ucenje modela za test.
                                                                                                           
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.501       7            42.33 draws/s   0:00:35   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.556       7            21.18 draws/s   0:01:10   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Test NLL: 16.549092
Ablaciona provera, priblizni blok-bootstrap: maxT=0/4 | BY=0/4.
Zavrsno ucenje: svi istorijski prelazi.
                                                                                                           
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.504       7            37.90 draws/s   0:00:39   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.627       7            18.82 draws/s   0:01:19   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
NEXT: deterministicka pretraga posteriorne aproksimacije sa 128 uzoraka parametara.
NEXT: [4, 6, 8, 15, 18, 26, 38]



CSV: loto7_4724_k93_loto_2984.csv
Izmereno racunanje likelihood-a i gradijenta: 1.44 ms | Numba niti: 4.
Ucenje posteriora: prior=0.35.
                                                                                                           
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.567       3            86.47 draws/s   0:00:17   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.637       7            43.50 draws/s   0:00:34   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Ucenje posteriora: prior=1.
                                                                                                          
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.527       7            84.81 draws/s   0:00:17   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.654       7            42.83 draws/s   0:00:35   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Validacija NLL: 16.549195 | prior: 0.35
Ucenje modela za test.
                                                                                                           
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.546       7            70.26 draws/s   0:00:21   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.641       7            35.45 draws/s   0:00:42   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Test NLL: 16.547972
Ablaciona provera, priblizni blok-bootstrap: maxT=0/4 | BY=0/4.
Zavrsno ucenje: svi istorijski prelazi.
                                                                                                           
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.556       3            52.16 draws/s   0:00:28   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.644       7            27.42 draws/s   0:00:54   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
NEXT: deterministicka pretraga posteriorne aproksimacije sa 128 uzoraka parametara.
NEXT: [2, 13, 16, 17, 25, 29, 36]



CSV: loto7_4724_k93_loto_plus_1740.csv
Izmereno racunanje likelihood-a i gradijenta: 0.97 ms | Numba niti: 4.
Ucenje posteriora: prior=0.35.
                                                                                                           
  Progress              Draw   Divergences   Step size   Grad evals   Speed            Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━   1500   0             0.571       7            134.82 draws/s   0:00:11   0:00:00    
  ━━━━━━━━━━━━━━━━━━━   1500   0             0.653       7            67.13 draws/s    0:00:22   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Ucenje posteriora: prior=1.
                                                                                                           
  Progress              Draw   Divergences   Step size   Grad evals   Speed            Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━   1500   0             0.558       3            136.98 draws/s   0:00:10   0:00:00    
  ━━━━━━━━━━━━━━━━━━━   1500   0             0.618       7            68.29 draws/s    0:00:21   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Validacija NLL: 16.555033 | prior: 0.35
Ucenje modela za test.
                                                                                                           
  Progress              Draw   Divergences   Step size   Grad evals   Speed            Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━   1500   0             0.548       7            115.33 draws/s   0:00:13   0:00:00    
  ━━━━━━━━━━━━━━━━━━━   1500   0             0.589       7            57.64 draws/s    0:00:26   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
Test NLL: 16.549129
Ablaciona provera, priblizni blok-bootstrap: maxT=1/4 | BY=0/4.
Zavrsno ucenje: svi istorijski prelazi.
                                                                                                           
  Progress               Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.526       7            94.48 draws/s   0:00:15   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━   1500   0             0.691       7            48.18 draws/s   0:00:31   0:00:00    
                                                                                                            
Uzorkovanje zavrseno. Provera Rhat, ESS i BFMI.
Dijagnostika izracunata.
NEXT: deterministicka pretraga posteriorne aproksimacije sa 128 uzoraka parametara.
NEXT: [1, 7, 11, 25, 26, 33, 37]
"""



"""
NUTS za procenu neizvesnosti naučenih parametara, uz kontrolu višestrukog testiranja radi provere lažnih obrazaca.

NUTS procenjuje parametre modela, a NEXT je deterministički maksimum zajedničke prediktivne raspodele. 
Provere višestrukog testiranja je dijagnostika, bez prepravljanja kombinacije.

Kvantne amplitude daju verovatnoću cele sedmorke, a NUTS procenjuje neizvesnost parametara. 
Provera doprinosa koristiće blok-bootstrap i korekcije višestrukog testiranja, bez biranja modela na test-skupu. 

V1 povezuje NUTS, kvantne amplitude i proveru doprinosa pomoću blok-bootstrap postupka i korekcija višestrukog testiranja. 
NEXT je deterministički maksimum aproksimirane posteriorne prediktivne raspodele, bez nasumičnog generisanja kombinacija.

NUTS je zasnovan na radu Hoffmana i Gelmana. 
Bootstrap provera je adaptacija za vremenske podatke, ne tvrdnja o bezuslovnoj primeni Westfall-Young garancija. 

Potrebni su PyMC 5, ArviZ, PyTensor, NumPy, SciPy, Numba i Qiskit. 


Likelihood i analitički gradijent računaju se zajedno, samo za važeće kandidate, u paralelnoj Numba funkciji. 
NUTS koristi punu adaptacionu matricu za četiri parametra. 
Ugrađena je provera jednakosti sa prethodnom formulom. 
Takvo povezivanje spoljnog gradijenta podržava PyMC.


Za svaki CSV postoje četiri učenja:
1. Kandidat prior=0.35.
2. Kandidat prior=1.
3. Model za test.
4. Završni model na celoj istoriji, zatim NEXT.
"""





"""
The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
Journal of Machine Learning Research. The sampler PyMC uses by default, and the method that mapped the curved ridge in the fits to real draws.
Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm that avoids the random walk behavior and sensitivity to correlated parameters that plague many MCMC methods by taking a series of steps informed by first-order gradient information. These features allow it to converge to high-dimensional target distributions much more quickly than simpler methods such as random walk Metropolis or Gibbs sampling. However, HMC's performance is highly sensitive to two user-specified parameters: a step size ϵ
 and a desired number of steps L
. In particular, if L
 is too small then the algorithm exhibits undesirable random walk behavior, while if L
 is too large the algorithm wastes computation. We introduce the No-U-Turn Sampler (NUTS), an extension to HMC that eliminates the need to set a number of steps L
. NUTS uses a recursive algorithm to build a set of likely candidate points that spans a wide swath of the target distribution, stopping automatically when it starts to double back and retrace its steps. Empirically, NUTS performs at least as efficiently as (and sometimes more efficiently than) a well tuned standard HMC method, without requiring user intervention or costly tuning runs. We also derive a method for adapting the step size parameter ϵ
 on the fly based on primal-dual averaging. NUTS can thus be used with no hand-tuning at all, making it suitable for applications such as BUGS-style automatic inference engines that require efficient 
"""
