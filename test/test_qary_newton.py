"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
import tempfile
import unittest

QS=[2,3,5,9,27]
SHAPES = [(1000,), (32, 32), (1000, 1000), (4, 4, 8, 8)]
ROHS_DIST = [(10.0, 2.0), (1.0, 2.0), (1e-5, 1e-6)]

def createPLSCases(alphas, shapes, qs, rhos_dists):
    cases = []
    for alpha in alphas:
        for q in qs:
            for shape in shapes:
                for rhos_dist in rhos_dists:
                    n = int(np.prod(shape))
                    cases.append((cl.simulate._qary.PLSObjective(n=n, alpha=alpha), q, shape, rhos_dist))
    return cases

def createDLSCases(averageDistortions, shapes, qs, rhos_dists):
    cases = []
    for distortion in averageDistortions:
        for q in qs:
            for shape in shapes:
                for rhos_dist in rhos_dists:
                    n = int(np.prod(shape))
                    cases.append((cl.simulate._qary.DLSObjective(distortion=distortion, n=n), q, shape, rhos_dist))
    return cases


class TestSimulate(unittest.TestCase):
    """Test suite for simulate module."""
    """Tests to test multiple combinations of q,shape, and rho distribution parameters for both PLS and DLS objectives."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    def objectiveOutput(self,objective:cl.simulate._qary.PLSObjective):
        return f'PLS Alpha: {objective.alpha}', f'n: {objective.n}'
    
    def objectiveOutput(self,objective:cl.simulate._qary.DLSObjective):
        return f'DLS Distortion: {objective.distortion}'

    @parameterized.expand(
    createPLSCases(alphas=[.05, .1, .2, .4], shapes=SHAPES, qs=QS, rhos_dists=ROHS_DIST) +
    createDLSCases(averageDistortions=[0.1, 0.2, 0.3], shapes=SHAPES, qs=QS, rhos_dists=ROHS_DIST))
    def test_simulate_q_random(self, testCase: cl.simulate._qary.SenderObjective, q: int, shape: tuple, rhos_dist: tuple):
        self._logger.info(f'TestSimulate.test_simulate_q_random_{testCase.name}_q{q}')

        rng = np.random.default_rng(12345)
        rhos = tuple(
            np.maximum(rng.normal(rhos_dist[0], rhos_dist[1], shape), 1e-10)
            for _ in range(q - 1)
        )
    
        newTon = cl.simulate._qary.NewtonQarySimulation(sender=testCase)
        try:
            ps, lbda = newTon.probability(rhos=rhos)
            _, calcedTarget = testCase.objective(lbda=lbda, rhos=rhos)
         
            self.assertAlmostEqual(testCase.target(), calcedTarget, 3)
        except ValueError:
            print('Infeasible combination of parameters, skipping test')
            print(self.objectiveOutput(testCase), f'q: {q}', f'shape: {shape}', f'rhos_dist: {rhos_dist}')
            self.skipTest('infeasible combination')
        
    


__all__ = ['TestSimulate']
