from ._optim import Sender, average_distortion, average_payload
from typing import Tuple, Callable
import numpy as np
import logging

def avg_payload_derivative(
    ps: Tuple[np.ndarray] = None,
    rhos: Tuple[np.ndarray] = None
):
    """
    Derivative of the average payload (entropy H(beta)) w.r.t. lambda.
    Used by Newton's method for the PLS objective.

    :param ps: Tuple of q-1 arrays, ps[j][i] = probability of change direction j at element i.
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rhos: Tuple of q-1 arrays, rhos[j][i] = cost of change direction j at element i.
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Scalar derivative dH/dlambda.
    :rtype: float
    """
    e_rho = np.zeros_like(rhos[0])
    for p, rho in zip(ps, rhos):
        e_rho += p * rho

    ln_term = 1.0 / np.log(2)
    fp = 0.0

    for p, rho in zip(ps, rhos):
        middle_block = -rho + e_rho
        log_term = np.log2(p + 1e-10) + ln_term
        fp -= np.sum(p * middle_block * log_term)

    p0 = 1.0 - sum(ps)
    fp -= np.sum(p0 * e_rho * (np.log2(p0 + 1e-10) + ln_term))
    return fp


def avg_distortion_derivative(ps, rhos, **kw):
    """
    Derivative of the average distortion E[rho] w.r.t. lambda.
    Used by Newton's method for the DLS objective.

    :param ps: Tuple of q-1 arrays, ps[j][i] = probability of change direction j at element i.
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rhos: Tuple of q-1 arrays, rhos[j][i] = cost of change direction j at element i.
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Scalar derivative dE[rho]/dlambda.
    :rtype: float
    """
    e_rho = np.zeros_like(rhos[0])
    for p, rho in zip(ps, rhos):
        e_rho += p * rho

    fp = 0.0
    for p, rho in zip(ps, rhos):
        term1 = -p * (rho**2)
        term2 = rho * p * e_rho
        fp += np.sum(term1 + term2)
    return fp


class SenderObjective:
    """
    Wraps a sender type together with its objective function, derivative, and target value.
    Subclassed by PLSObjective and DLSObjective.

    :param sender: Sender enum value (PAYLOAD_LIMITED_SENDER or DISTORTION_LIMITED_SENDER).
    :type sender: Sender
    :param objective: Computes Gibbs probabilities and objective value.
    :type objective: Callable
    :param derivative: Derivative of objective w.r.t. lambda. None if not provided.
    :type derivative: Callable
    :param target: The target value Newton's method solves for (alpha*N for PLS, distortion for DLS).
    :type target: float
    """
    def __init__(self, sender: Sender, objective: Callable, derivative: Callable = None, target: float = None):
        self.sender = sender
        self.objective = objective
        self.derivative = derivative
        self.target_value = target
        self.name = f'{sender.name}'

    def target(self):
        """
        Returns the target value for the optimization.
        
        :return: Target value
        :rtype: float
        """
        return self.target_value


class PLSObjective(SenderObjective):
    """
    Payload-Limited Sender objective: find lambda s.t. H(beta) = alpha * N.

    :param n: Number of elements in the cover (e.g. pixels).
    :type n: float
    :param alpha: Embedding rate in bits per element. Target = n * alpha.
    :type alpha: float
    """
    def __init__(self, n: float = None, alpha: float = None):
        super().__init__(Sender.PAYLOAD_LIMITED_SENDER, average_payload, avg_payload_derivative, n * alpha)
        self.n = n
        self.alpha = alpha


class DLSObjective(SenderObjective):
    """
    Distortion-Limited Sender objective: find lambda s.t. E[rho] = distortion.

    :param distortion: Target average distortion per element.
    :type distortion: float
    :param n: Number of elements (used for logging/test output only).
    :type n: float
    """
    def __init__(self, distortion: float = None, n: float = None):
        super().__init__(Sender.DISTORTION_LIMITED_SENDER, average_distortion, avg_distortion_derivative, distortion)
        self.distortion = distortion
        self.n = n


class QarySimulation:
    """
    Base class for q-ary steganographic simulation.
    Subclasses implement calc_lambda to find the optimal Gibbs parameter.

    :param sender: A SenderObjective instance (PLSObjective or DLSObjective).
    :type sender: SenderObjective
    """
    def __init__(self, sender: SenderObjective):
        self.senderObjective = sender

    def quary(self,
              rhos: Tuple[np.ndarray],
              changes: list = None,
              q: float = 2,
              generator: str = None,
              order: str = 'C',
              seed: int = None,
              *args, **kwargs):
        """
        Full pipeline: compute optimal embedding probabilities and sample changes.

        :param rhos: Tuple of q-1 cost arrays, one per change direction.
        :type rhos: tuple
        :param changes: List of q-1 integer change values.
        :type changes: list
        :param q: Number of states per element (2=binary, 3=ternary, etc.).
        :type q: float
        :param generator: RNG type. None=numpy default, 'MT19937'=Matlab-compatible.
        :type generator: str
        :param order: Traversal order for random array. 'C'=row-major, 'F'=column-major.
        :type order: str
        :param seed: Random seed for reproducibility.
        :type seed: int
        :return: Integer array of sampled changes
        :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
        """
        if changes is None:
            logging.warning('No change values provided, using default values. Using symmetric costs +1,-1,+2,-2,...')
            changes = []
            for i in range(1, q):
                if i % 2 == 1:
                    changes.append((i + 1) // 2)
                else:
                    changes.append(-i // 2)
        elif len(changes) != q - 1:
            raise ValueError(f'Number of change values must be q-1, but got {len(changes)} change values for q={q}')

        ps, _ = self.probability(rhos=rhos, *args, **kwargs)
        return self.simulate(ps=ps, changes=changes, generator=generator, order=order, seed=seed, *args, **kwargs)

    def simulate(self, ps, changes, generator: str = None, order: str = 'C', seed: int = None, *args, **kwargs):
        """
        Sample embedding changes from the probability maps.

        :param ps: Tuple of q-1 probability arrays.
        :type ps: tuple
        :param changes: Tuple of q-1 integer change values corresponding to each direction.
        :type changes: tuple
        :param generator: RNG type (see quary).
        :type generator: str
        :param order: Array traversal order (see quary).
        :type order: str
        :param seed: Random seed.
        :type seed: int
        :return: Integer array of sampled changes
        :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
        """
        if generator is None:
            rng = np.random.default_rng(seed=seed)
            rand_change = rng.random(ps[0].shape)
        elif generator == 'MT19937':
            prng = np.random.RandomState(seed)
            rand_change = prng.random_sample(ps[0].shape)
        else:
            try:
                rand_change = generator(ps[0].shape)
            except Exception:
                raise NotImplementedError(f'unsupported generator {generator}')

        if order is None or order == 'C':
            pass
        elif order == 'F':
            rand_change = rand_change.reshape(-1).reshape(ps[0].shape, order='F')
        else:
            raise NotImplementedError(f'Given order {order} is not implemented')

        delta = np.zeros(ps[0].shape, dtype='int8')

        for i in range(ps[0].shape[0]):
            for j in range(ps[0].shape[1]):
                boundary = 0
                for change_value, p_map in zip(changes, ps):
                    if rand_change[i, j] < boundary + p_map[i, j]:
                        delta[i, j] = change_value
                        break
                    boundary += p_map[i, j]
        return delta

    def probability(self, rhos, *args, **kwargs):
        """
        Compute optimal embedding probabilities by finding lambda via calc_lambda.

        :param rhos: Tuple of q-1 cost arrays.
        :type rhos: tuple
        :return: Tuple of q-1 probability arrays and the optimal lambda value.
        :rtype: tuple
        """
        lbda = self.calc_lambda(rhos=rhos)
        ps, _ = self.senderObjective.objective(lbda=lbda, rhos=rhos)
        return ps, lbda

    def calc_lambda(self, rhos):
        pass


class NewtonQarySimulation(QarySimulation):
    """
    Q-ary simulation using Newton's method to find optimal lambda.
    Supports arbitrary q >= 2.

    :param sender: A SenderObjective instance (PLSObjective or DLSObjective).
    :type sender: SenderObjective
    :param max_iter: Maximum number of Newton iterations (default 50).
    :type max_iter: int
    :param xtol: Convergence tolerance for residual and lambda difference.
    :type xtol: float
    """
    def __init__(self, sender: SenderObjective, max_iter=50, xtol=1e-5):
        super().__init__(sender)
        self.max_iter = max_iter
        self.xtol = xtol

    def calc_lambda(self, rhos):
        """
        Delegates to Newton solver using the sender's target and objective.
        
        :param rhos: Tuple of cost arrays.
        :type rhos: tuple
        :return: Optimal lambda value.
        :rtype: float
        """
        return self.calc_lambda_newton(
            rhos=rhos,
            target=self.senderObjective.target(),
            objective=self.senderObjective.objective
        )

    def exponential_lambda_search(self, rho, target, objective):
        """
        Find an initial bracket [lbd_l, lbd_r] containing the root via exponential expansion.

        :param rho: Tuple of cost arrays.
        :type rho: tuple
        :param target: Target value to bracket.
        :type target: float
        :param objective: Callable (lbda, rhos) -> (ps, value).
        :type objective: Callable
        :return: Bracket such that objective(lbd_l) > target >= objective(lbd_r).
        :rtype: tuple
        """
        lbd_l = 1e-30
        lbd_r = 1e-4
        for _ in range(15):
            _, v = objective(lbda=lbd_r, rhos=rho)
            if v > target:
                lbd_l = lbd_r
                lbd_r *= 10
            else:
                break
        return lbd_l, lbd_r

    def calc_lambda_newton(self, rhos: Tuple[np.ndarray], target: float, objective: Callable):
        """
        Newton's method to find lambda.

        :param rhos: Tuple of q-1 cost arrays.
        :type rhos: tuple
        :param target: Target value (alpha*N for PLS, distortion for DLS).
        :type target: float
        :param objective: Callable (lbda, rhos) -> (ps, value).
        :type objective: Callable
        :return: Converged lambda value.
        :rtype: float
        :raises ValueError: If target is outside the achievable range.
        :raises NotImplementedError: If no derivative function is set.
        """
        _, v_max = objective(lbda=1e-30, rhos=rhos)
        _, v_min = objective(lbda=1e30, rhos=rhos)

        if target < v_min or target > v_max:
            raise ValueError(f'Target value {target} is out of bounds [{v_min}, {v_max}] for the given rhos')

        lbdl, lbdrr = self.exponential_lambda_search(rhos, target, objective)
        lbd = (lbdl + lbdrr) / 2
        last_lbd = lbd

        for _ in range(self.max_iter):
            ps, v = objective(lbda=lbd, rhos=rhos)
            f = v - target

            logging.info(f'lambda: {lbd}, f: {abs(f)}')
            if self.senderObjective.derivative is None:
                raise NotImplementedError('Derivative function not implemented for Newton\'s method')

            f_prime = self.senderObjective.derivative(ps=ps, rhos=rhos)

            if abs(f) < self.xtol:
                break

            if abs(f_prime) < 1e-20:
                break

            lbd -= f / f_prime
            lbd = np.clip(lbd, lbdl, lbdrr)

            if abs(last_lbd - lbd) < self.xtol:
                break

            last_lbd = lbd

        return lbd
