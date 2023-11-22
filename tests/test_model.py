"""
Test functions for the model itself.

First, a fixture is defined.
The fixture will be run some timesteps in test_run_spinup()
All following tests simulate running a single timestep 
method by method, thereby checking their correct functionality 
individually.

Runtime variables that are not model-attributes are stored in
the ValueStorage class that is set up for this purpose.

TODO: 
- most test methods don't feature any assertions/sanity-checks yet,
  add some sanity checks to actually make them test something!
"""

import pytest

import numpy as np
import matplotlib.pyplot as plt

from mayasim.model.core import Core as Model
from .conftest import ValueStorage as val

###############################################################################
# model instance fixture
###############################################################################


@pytest.fixture(scope="session", name="model_instance")
def model_instance_fixture():
    '''
    TODO: cover more parameter settings
    '''
    test_path = 'output/test_model/'

    return Model(output_path=test_path)


###############################################################################
# spinup fixture: test run()
###############################################################################


@pytest.mark.dependency()
def test_run_spinup(model_instance):
    '''TODO: add sanity checks'''
    # run model
    model_instance.run(t_max = val.spinup)

    # set test-internal time counter
    val.t = val.spinup + 1


###############################################################################
# test run-methods separately
###############################################################################

# ecosystem ###################################################################

@pytest.mark.dependency(depends=['test_run_spinup'])
def test_update_precipitation(model_instance):
    '''TODO: add sanity checks'''
    model_instance.update_precipitation(val.t)


@pytest.mark.dependency(depends=['test_update_precipitation'])
def test_get_npp(model_instance):
    '''TODO: add sanity checks'''
    val.npp = model_instance.get_npp()


@pytest.mark.dependency(depends=['test_get_npp'])
def test_evolve_forest(model_instance):
    '''TODO: add sanity checks'''
    model_instance.evolve_forest(val.npp)


@pytest.mark.dependency(depends=['test_evolve_forest'])
def test_get_waterflow(model_instance):
    '''TODO: add sanity checks'''
    val.wf = model_instance.get_waterflow()[1]


@pytest.mark.dependency(depends=['test_get_waterflow'])
def test_get_ag(model_instance):
    '''TODO: add sanity checks'''
    val.ag = model_instance.get_ag(val.npp, val.wf)


@pytest.mark.dependency(depends=['test_get_ag'])
def test_get_ecoserv(model_instance):
    '''TODO: add sanity checks'''
    val.es = model_instance.get_ecoserv(val.ag, val.wf)


# society #####################################################################

# ag income

@pytest.mark.dependency(depends=['test_get_ecoserv'])
def test_benefit_cost(model_instance):
    '''TODO: add sanity checks'''
    val.bca = model_instance.benefit_cost(val.ag)


@pytest.mark.dependency(depends=['test_benefit_cost'])
def test_get_influenced_cells(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_influenced_cells()


@pytest.mark.dependency(depends=['test_get_influenced_cells'])
def test_get_cropped_cells(model_instance):
    '''
    TODO: check for correct addition and deletion of cropped cells
    '''
    val.abandoned, val.sown = \
        model_instance.get_cropped_cells(val.bca)


@pytest.mark.dependency(depends=['test_get_cropped_cells'])
def test_get_crop_income(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_cropped_cells(val.bca)


# es income

@pytest.mark.dependency(depends=['test_get_crop_income'])
def test_get_eco_income(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_eco_income(val.es)


@pytest.mark.dependency(depends=['test_get_eco_income'])
def test_evolve_soil_deg(model_instance):
    '''TODO: add sanity checks'''
    model_instance.evolve_soil_deg()


@pytest.mark.dependency(depends=['test_evolve_soil_deg'])
def test_update_pop_gradient(model_instance):
    '''TODO: add sanity checks'''
    model_instance.update_pop_gradient()


# trade income

@pytest.mark.dependency(depends=['test_update_pop_gradient'])
def test_get_rank(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_rank()


@pytest.mark.dependency(depends=['test_get_rank'])
def test_build_routes(model_instance):
    '''TODO: add sanity checks'''
    (val.built, val.lost) = model_instance.build_routes


@pytest.mark.dependency(depends=['test_build_routes'])
def test_get_comps(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_comps()


@pytest.mark.dependency(depends=['test_get_comps'])
def test_get_centrality(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_centrality()


@pytest.mark.dependency(depends=['test_get_comps'])
def test_get_trade_income(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_trade_income()


# total income

@pytest.mark.dependency(depends=['test_get_trade_income'])
def test_get_real_income_pc(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_real_income_pc()


# migration

@pytest.mark.dependency(depends=['test_get_real_income_pc'])
def test_get_pop_mig(model_instance):
    '''TODO: add sanity checks'''
    model_instance.get_pop_mig()


@pytest.mark.dependency(depends=['test_get_pop_mig'])
def test_migration(model_instance):
    '''TODO: add sanity checks'''
    val.new_settlements = model_instance.migration(val.es)


@pytest.mark.dependency(depends=['test_migration'])
def test_kill_settlements(model_instance):
    '''TODO: add sanity checks'''
    val.killed_settlements = model_instance.kill_settlements()


# output ######################################################################

@pytest.mark.dependency(depends=['test_kill_settlements'])
def test_step_output(model_instance):
    '''TODO: add sanity checks'''

    # get settlement attributes
    stm_attr_names = [n for n in dir(model_instance) if n.startswith('stm_')]
    stm_attr_names.remove('stm_positions') # has different first dimension
    stm_attr = [getattr(model_instance, n) for n in stm_attr_names]

    # assert length of settlemnt attributes
    assert np.all([len(a) == model_instance.n_settlements for a in stm_attr])

    model_instance.step_output(
        val.t, val.npp, val.wf, val.ag, val.es, val.bca,
        val.abandoned, val.sown, val.built, val.lost,
        val.new_settlements, val.killed_settlements)


@pytest.mark.dependency(depends=['test_step_output'])
def test_aggregates(model_instance):
    """
    save a plot of some aggregate values to
    'MayaSim/output/test_model/aggregates_plot.png'
    TODO: add sanity checks
    """

    # get aggregates
    aggs = model_instance.get_aggregates()
    # select some measures, include time dimension
    measures = [
        'time',
        'total_population',
        'total_settlements',
        'total_migrants'
        ]

    # plot aggregates
    fig, axes = plt.subplots(ncols=len(measures)-1, figsize=(16, 2))
    for i, meas in enumerate(measures[1:]):
        aggs.plot('time', y=meas, ax=axes[i], title=meas)

    fig.savefig('output/test_model/aggregates_plot.png')

    assert aggs.shape[0] == int(aggs.at[aggs.index[-1],'time'])
