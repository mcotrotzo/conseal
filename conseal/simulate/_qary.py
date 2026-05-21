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

    Args:
        ps:   Tuple of q-1 arrays, ps[j][i] = probability of change direction j at element i.
        rhos: Tuple of q-1 arrays, rhos[j][i] = cost of change direction j at element i.

    Returns:
        fp: Scalar derivative dH/dlambda.
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

    Args:
        ps:   Tuple of q-1 arrays, ps[j][i] = probability of change direction j at element i.
        rhos: Tuple of q-1 arrays, rhos[j][i] = cost of change direction j at element i.

    Returns:
        fp: Scalar derivative dE[rho]/dlambda.
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

    Args:
        sender:     Sender enum value (PAYLOAD_LIMITED_SENDER or DISTORTION_LIMITED_SENDER).
        objective:  Callable (lbda, rhos) -> (ps, value). Computes Gibbs probabilities and objective value.
        derivative: Callable (ps, rhos) -> scalar. Derivative of objective w.r.t. lambda. None if not provided.
        target:     The target value Newton's method solves for (alpha*N for PLS, distortion for DLS).
    """
    def __init__(self, sender: Sender, objective: Callable, derivative: Callable = None, target: float = None):
        self.sender = sender
        self.objective = objective
        self.derivative = derivative
        self.target_value = target
        self.name = f'{sender.name}'

    def target(self):
        """Returns the target value for the optimization."""
        return self.target_value


class PLSObjective(SenderObjective):
    """
    Payload-Limited Sender objective: find lambda s.t. H(beta) = alpha * N.

    Args:
        n:     Number of elements in the cover (e.g. pixels).
        alpha: Embedding rate in bits per element. Target = n * alpha.
    """
    def __init__(self, n: float = None, alpha: float = None):
        super().__init__(Sender.PAYLOAD_LIMITED_SENDER, average_payload, avg_payload_derivative, n * alpha)
        self.n = n
        self.alpha = alpha


class DLSObjective(SenderObjective):
    """
    Distortion-Limited Sender objective: find lambda s.t. E[rho] = distortion.

    Args:
        distortion: Target average distortion per element.
        n:          Number of elements (used for logging/test output only).
    """
    def __init__(self, distortion: float = None, n: float = None):
        super().__init__(Sender.DISTORTION_LIMITED_SENDER, average_distortion, avg_distortion_derivative, distortion)
        self.distortion = distortion
        self.n = n


class QarySimulation:
    """
    Base class for q-ary steganographic simulation.
    Subclasses implement calc_lambda to find the optimal Gibbs parameter.

    Args:
        sender: A SenderObjective instance (PLSObjective or DLSObjective).
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

        Args:
            rhos:      Tuple of q-1 cost arrays, one per change direction.
            changes:   List of q-1 integer change values (e.g. +1,-1 ).
                       If None, defaults to symmetric values +1,-1,+2,-2,...
            q:         Number of states per element (2=binary, 3=ternary, etc.).
            generator: RNG type. None=numpy default, 'MT19937'=Matlab-compatible. 
                       Can also be a callable (shape) -> array.
            order:     Traversal order for random array. 'C'=row-major, 'F'=column-major.
            seed:      Random seed for reproducibility.

        Returns:
            delta: Integer array of sampled changes
        """
        if changes is None:
            logging.warning('No change values provided, using default values. Using symmetric costs +1,-1,+2,-2,...')
            changes = []
            # Generate q-1 symmetric change values
            for i in range(1, q):
                # Odd index -> positive (+1, +2, ...)
                if i % 2 == 1:
                    changes.append((i + 1) // 2)
                else:
                    # Even index -> negative (-1, -2, ...)
                    changes.append(-i // 2)
        elif len(changes) != q - 1:
            raise ValueError(f'Number of change values must be q-1, but got {len(changes)} change values for q={q}')

        ps, _ = self.probability(rhos=rhos, *args, **kwargs)
        return self.simulate(ps=ps, changes=changes, generator=generator, order=order, seed=seed, *args, **kwargs)

    def simulate(self, ps, changes, generator: str = None, order: str = 'C', seed: int = None, *args, **kwargs):
        """
        Sample embedding changes from the probability maps.

        Args:
            ps:        Tuple of q-1 probability arrays, ps[j][i] = P(change direction j at element i).
            changes:   Tuple of q-1 integer change values corresponding to each direction.
            generator: RNG type (see quary).
            order:     Array traversal order (see quary).
            seed:      Random seed.

        Returns:
            delta: Integer array of sampled changes
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

        #calc change vector
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

        Args:
            rhos: Tuple of q-1 cost arrays.

        Returns:
            ps:   Tuple of q-1 probability arrays under the optimal Gibbs distribution.
            lbda: The optimal lambda value found.
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

    Args:
        sender:   A SenderObjective instance (PLSObjective or DLSObjective).
        max_iter: Maximum number of Newton iterations (default 50).
        xtol:     Convergence tolerance for residual and previous lambda difference.
    """
    def __init__(self, sender: SenderObjective, max_iter=50, xtol=1e-5):
        super().__init__(sender)
        self.max_iter = max_iter
        self.xtol = xtol

    def calc_lambda(self, rhos):
        """Delegates to Newton solver using the sender's target and objective."""
        return self.calc_lambda_newton(
            rhos=rhos,
            target=self.senderObjective.target(),
            objective=self.senderObjective.objective
        )

    def exponential_lambda_search(self, rho, target, objective):
        """
        Find an initial bracket [lbd_l, lbd_r] containing the root via exponential expansion.
        Starts from lbd_r=1e-4 and multiplies by 10 until objective(lbd_r) <= target.

        Args:
            rho:       Tuple of cost arrays.
            target:    Target value to bracket.
            objective: Callable (lbda, rhos) -> (ps, value).

        Returns:
            (lbd_l, lbd_r): Bracket such that objective(lbd_l) > target >= objective(lbd_r).
        """
        # start as low as possible
        lbd_l = 1e-30
        # Initial upper bound candidate
        lbd_r = 1e-4
        for _ in range(15):
            _, v = objective(lbda=lbd_r, rhos=rho)
            if v > target:
                lbd_l = lbd_r
                lbd_r *= 10
            else:
                break
        return lbd_l, lbd_r

    def calc_lambda_newton(self,
                           rhos: Tuple[np.ndarray],
                           target: float,
                           objective: Callable):
        """
        Newton's method to find lambda.
        Args:
            rhos:      Tuple of q-1 cost arrays.
            target:    Target value (alpha*N for PLS, distortion for DLS).
            objective: Callable (lbda, rhos) -> (ps, value).
        Returns:
            lbd: Converged lambda value.
        Raises:
            ValueError: If target is outside the achievable range [v_min, v_max] for the given rhos.
            NotImplementedError: If no derivative function is set on the sender objective.
        """
        _, v_max = objective(lbda=1e-30, rhos=rhos)  # Maximum achievable value (lambda -> 0)
        _, v_min = objective(lbda=1e30, rhos=rhos)   # Minimum achievable value (lambda -> inf)

        # this i do espevially for the DLS objective, when the user selects a distortion target that is not achievable.
        if target < v_min or target > v_max:
            raise ValueError(f'Target value {target} is out of bounds [{v_min}, {v_max}] for the given rhos')

        #thirst i use a first exponential search to find a good bracket
        lbdl, lbdrr = self.exponential_lambda_search(rhos, target, objective)
        lbd = (lbdl + lbdrr) / 2
        last_lbd = lbd

        for _ in range(self.max_iter):
            #gets betas and calcs the current objective value for the current lambda
            ps, v = objective(lbda=lbd, rhos=rhos)
            #residual between current objective value and target
            f = v - target

            logging.info(f'lambda: {lbd}, f: {abs(f)}')
            # this prevents that the user uses a sender objective without derivative function implemented, which is required for newton's method
            if self.senderObjective.derivative is None:
                raise NotImplementedError('Derivative function not implemented for Newton\'s method')

            #calculates derivative
            f_prime = self.senderObjective.derivative(ps=ps, rhos=rhos)

            #checks if converged
            if abs(f) < self.xtol:
                break
            

            # Prevent division by zero small derivative
            if abs(f_prime) < 1e-20:
                break
            
            #the newton step
            lbd -= f / f_prime             

            #prevents overshoot
            lbd = np.clip(lbd, lbdl, lbdrr)

            #prevents stuck in the convergence
            if abs(last_lbd - lbd) < self.xtol:
                break

            last_lbd = lbd

        return lbd