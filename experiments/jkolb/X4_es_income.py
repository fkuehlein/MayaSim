"""
I want to know, why the income from ecosystem services is so robust against the
deterioration of the ecosystem.
Therefore, run the model without killing settlements without agriculture and
maybe mess with the parameters for different sources of income from
ecosystem services.
"""

# disable pylint invalid-name message on module level
# pylint: disable=invalid-name
# pylint: enable=invalid-name

import sys
import getpass

import itertools as it
import pickle as pkl
import numpy as np
import pandas as pd

from pymofa.experiment_handling import experiment_handling as eh
from mayasim.model.core import Core as Model
from mayasim.model.parameters import Parameters

TEST = True


def run_function(n_init=30, kill_cropless=False, better_ess=False,
                 steps=350, filename='./'):
    """
    Set up the Model for default Parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values and Parameters, as well as the time
    development of aggregated variables for each run.

    Parameters:
    -----------
    n_init : int > 0
        initial number of settlements on the map,
    kill_cropless: bool
        switch to either kill settlements without crops or not
    better_es: bool
        switch for more realistic income from ecosystem services
    steps: int
        number of steps to integrate the model for,
    filename: string
        path to save the results to.
    """

    # initialize the Model
    m = Model(n_init, output_path=filename)

    m.kill_cities_without_crops = kill_cropless
    m.better_ess = better_ess

    if not filename.endswith('s0.pkl'):
        m.output_geographic_data = False
        m.output_settlement_data = False

    # store initial conditions and Parameters
    res = {
        "initials": pd.DataFrame({
            "Settlement X Positions": m.stm_positions[0],
            "Settlement Y Positions": m.stm_positions[1],
            "Population": m.stm_population
        }),
        "Parameters": pd.Series({
            key: getattr(m, key)
            for key in dir(Parameters)
            if not key.startswith('__')
            and not callable(key)
        })
    }

    # run Model
    if TEST:
        steps = 3

    m.run(steps)

    # Retrieve results
    res["aggregates"] = m.get_aggregates()
    res["traders aggregates"] = m.get_traders_aggregates()

    try:
        with open(filename, 'wb') as dumpfile:
            pkl.dump(res, dumpfile)
            return 1
    except IOError:
        return -1


# pylint: disable=too-many-locals
def run_experiment(argv):
    """
    Take argv input variables and run experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to sub_experiment
        data for post processing,
    5)  run computation and/or post processing and/or plotting
        depending on execution on cluster or locally or depending on
        experimentation mode.

    Parameters
    ----------
    argv: list
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if successful.
    """

    # Parse test switch from input
    global TEST  # pylint: disable=global-statement
    if __name__ == '__main__':
        TEST = len(argv) > 1 and argv[1] == 'test'
    else:
        TEST = argv == 'test'

    # Generate paths according to switches and user name
    test_folder = ['', 'test_experiments/'][int(TEST)]
    experiment_folder = 'X4_es_income/'
    raw = 'raw_data/'
    res = 'results/'

    if getpass.getuser() == 'fritz':
        save_path_raw = '/Users/fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{test_folder}{experiment_folder}{raw}'
        save_path_res = '/Users/fritz/Desktop/Thesis/MayaSim/' \
                        f'output/{test_folder}{experiment_folder}{res}'
    else:
        save_path_raw = f'./output/{test_folder}{experiment_folder}{raw}'
        save_path_res = f'./output/{test_folder}{experiment_folder}{res}'

    # Generate parameter combinations

    index = {0: "kill_cropless", 1: "better_ess"}

    kill_cropless = [True, False]
    better_ess = [True, False]

    param_combs = list(it.product(kill_cropless, better_ess))

    sample_size = 10 if not TEST else 2

    # Define names and callables for post processing

    name1 = "aggregates"
    estimators1 = {
        "mean_aggregates": lambda fnames: pd.concat([
            np.load(f, allow_pickle=True)["aggregates"]
            for f in fnames
        ]).groupby(level=0).mean(),
        "sigma_aggregates": lambda fnames: pd.concat([
            np.load(f, allow_pickle=True)["aggregates"]
            for f in fnames
        ]).groupby(level=0).std()
    }

    name2 = "traders_aggregates"
    estimators2 = {
        "mean_aggregates": lambda fnames: pd.concat([
            np.load(f, allow_pickle=True)["traders aggregates"]
            for f in fnames
        ]).groupby(level=0).mean(),
        "sigma_aggregates": lambda fnames: pd.concat([
            np.load(f, allow_pickle=True)["traders aggregates"]
            for f in fnames
        ]).groupby(level=0).std()
    }

    # Run computation and post processing.
    if TEST:
        print(f'testing {experiment_folder[:-1]}')
    handle = eh(sample_size=sample_size,
                parameter_combinations=param_combs,
                index=index,
                path_raw=save_path_raw,
                path_res=save_path_res,
                use_kwargs=True)

    handle.compute(run_func=run_function)
    handle.resave(eva=estimators1, name=name1)
    handle.resave(eva=estimators2, name=name2)

    return 1


if __name__ == '__main__':

    run_experiment(sys.argv)
