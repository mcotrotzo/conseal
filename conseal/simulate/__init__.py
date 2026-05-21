"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple

from . import _binary
from . import _qary
from ._qary import DLSObjective,PLSObjective
from . import _ternary
from ._binary import binary
from ._ternary import ternary

from ._optim import get_p, calc_lambda, average_payload,average_distortion
from ._optim import Sender, average_distortion

PAYLOAD_LIMITED_SENDER = Sender.PAYLOAD_LIMITED_SENDER
DISTORTION_LIMITED_SENDER = Sender.DISTORTION_LIMITED_SENDER
PLS = PAYLOAD_LIMITED_SENDER
DLS = DISTORTION_LIMITED_SENDER


def simulate(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    seed: int = None,
    q: int = None,
    **kw,
) -> Tuple[np.ndarray]:
    """

    :param rhos: either
        a distortion tensor for +-1 change, or
        a tuple with tensors for +1 and -1 change
        or a tuple with tensors for q changes
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :param n: Cover size.
    :type n: int
    :param seed: random seed for embedding simulator
    :type seed: int
    :return:
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y += cl.simulate.ternary(
    ...     rhos=rhos,        # costs of change
    ...     alpha=0.4,        # alpha
    ...     n=im_dct.Y.size,  # cover size
    ...     seed=12345)       # seed
    """
    # derive q if not given
    if q is None:
        q = len(rhos) + 1

    # ternary
    if q == 3:
        return _ternary.ternary(
            rhos=rhos,
            alpha=alpha,
            n=n,
            seed=seed,
            **kw,
        )
    elif q == 2:
        return _binary.binary(
            rhos=rhos,
            alpha=alpha,
            n=n,
            seed=seed,
            **kw,
        )
    # other
    else:
        raise NotImplementedError(f'Only q=2 and q=3 are implemented, but got q={q}')

def simulate_qary_newton(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    sender: Sender = PAYLOAD_LIMITED_SENDER,
    seed: int = None,
    q: int = None,
    distortionTarget: float = None,
    changes: list = None,
    **kw,
) -> Tuple[np.ndarray]:
    """
    Args:
    rhos: Tuple of q-1 cost tensors, rhos[j][i] = cost of change direction j at element i.
    alpha: Embedding rate (bits per element).
    n: Cover size (number of elements).
    sender: Sender typ -> PAYLOAD_LIMITED_SENDER,DISTORTION_LIMITED_SENDER etc.
    seed: Random seed for embedding simulator.
    q: Number of change directions (no-change is implicit included) -> q-1 changes.
    distortionTarget: Target average distortion for DISTORTION_LIMITED_SENDER.
    changes: List of q-1 change values, changes[j] = change value for change direction j. If None, defaults to symmetric changes +1,-1,+2,-2,...
    
    Returns:
    delta: Integer array of sampled changes
    """
    
    if sender == PAYLOAD_LIMITED_SENDER:
        senderObjective = PLSObjective(n=n, alpha=alpha)
    elif sender == DISTORTION_LIMITED_SENDER:
        if distortionTarget is None:
            raise ValueError('distortionTarget must be provided for DISTORTION_LIMITED_SENDER')
        senderObjective = DLSObjective(distortion=distortionTarget, n=n)
    else:
        raise ValueError(f'Unknown sender type: {sender}')
    
    newTonSimulator = _qary.NewtonQarySimulation(sender=senderObjective)

    return newTonSimulator.quary(rhos=rhos,seed=seed, q=q, changes=changes, **kw)
    

__all__ = [
    '_optim',
    '_ternary',
    'ternary',
    '_binary',
    'binary',
    'get_p',
    'simulate',
    'average_payload',
    'average_distortion',
    'calc_lambda',
    'Sender',
    'PAYLOAD_LIMITED_SENDER',
    'DISTORTION_LIMITED_SENDER',
    'DLS',
    'PLS',
    '_qary',               
    'simulate_qary_newton',
    'PLSObjective',        
    'DLSObjective',
]
