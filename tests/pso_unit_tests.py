import os
import sys
import numpy as np
import pytest


# Ensure repo root is on sys.path so `templates` is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import the Metaheuristic class from templates
from templates.metaheuristic import Metaheuristic


def get_instance_path():
	# Prefer instances/instance_test.json; fallback to a small provided instance
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	candidate = os.path.join(repo_root, 'instances', 'instance_test.json')
	if not os.path.exists(candidate):
		pytest.skip('instances/instance_test.json not found; skipping tests that require it')
	return candidate


def test_read_problem_instance_loads_attributes():
	instance_path = get_instance_path()
	met = Metaheuristic(time_deadline=1, problem_path=instance_path)
	met.read_problem_instance(instance_path)
	assert isinstance(met.n, int) and met.n > 0
	assert isinstance(met.k, int) and met.k > 0
	assert isinstance(met.R, (int, float))
	assert isinstance(met.r, np.ndarray) and met.r.shape == (met.n,)
	assert isinstance(met.d, np.ndarray) and met.d.shape == (met.n, met.n)


def test_initialize_population_shapes_and_types():
	instance_path = get_instance_path()
	met = Metaheuristic(time_deadline=1, problem_path=instance_path)
	met.read_problem_instance(instance_path)
	pop_size = 4
	population, velocity = met.initialize_population(pop_size)
	assert population.shape == (pop_size, met.n * 2)
	assert velocity.shape == (pop_size, met.n * 2)
	assert population.dtype == int
	assert velocity.dtype == float


def test_rate_returns_expected_shape_and_values():
	instance_path = get_instance_path()
	met = Metaheuristic(time_deadline=1, problem_path=instance_path)
	met.read_problem_instance(instance_path)

	pop_size = 4
	# Construct a simple valid population: positions + picks
	population = np.array([
        [100, 200, 300, 200, 0, 0, 0, 0], # Test no picks (numerical stability)
        [500, 10, 500, 200, 1, 0, 1, 0], # Test two assets picked
        [200, 200, 200, 10000, 1, 1, 1, 0], # Test three assets picked
        [1, 200, 999, 10000, 1, 0, 1, 0], # Test four small weight (numerical stability)
    ])

	rates = met.get_rates(population)
	rates_slice = met.get_rates(population[1:])
	rates_single = met.get_rates(population[1:2])
	assert isinstance(rates, np.ndarray)
	assert rates.shape == (pop_size,)
	assert rates_slice.shape == (pop_size - 1,)
	# Numeric sanity: rates should be finite
	assert np.all(np.isfinite(rates))
	third_sq = (1/3)**2
	expected_rates = [
		0, 
		0.89*0.5, 
		0.89*third_sq*2+0.55*third_sq*2+0.33*third_sq*2,
		0.999*0.001*0.89*2
	]
	np.testing.assert_allclose(rates, expected_rates, rtol=1e-5)
	np.testing.assert_array_equal(rates[1:], rates_slice)
	assert rates[1] == rates_single