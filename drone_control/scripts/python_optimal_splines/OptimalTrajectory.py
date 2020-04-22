from OptimalSplineGen import Waypoint
import OptimalSplineGen
import numpy as np
import itertools
import scipy.optimize
from math import factorial


class TrajectoryWaypoint:
    def __init__(self, ndim):
        self.time = None
        if isinstance(ndim, int):
            self.ndim = ndim
            self.spline_pins = [Waypoint(None) for i in range(self.ndim)]
        elif isinstance(ndim, tuple):
            self.ndim = len(ndim)
            self.spline_pins = [Waypoint(None) for i in range(self.ndim)]
            self.add_hard_constraints(0, ndim)

    def add_hard_constraints(self, order, values):
        assert len(values) == self.ndim
        for i, v in enumerate(values):
            self.spline_pins[i].add_hard_constraint(order, v)

    def add_soft_constraints(self, order, values, radii):
        assert len(values) == self.ndim
        for i, v in enumerate(values):
            self.spline_pins[i].add_soft_constraint(order, v, radii[i])

    def add_hard_constraint(self, order, dim, value):
        self.spline_pins[dim].add_hard_constraint(order, value)

    def add_soft_constraint(self, order, dim, value, radius):
        self.spline_pins[dim].add_soft_constraint(order, value, radius)

    def set_time(self, t):
        self.time = t
        for sp in self.spline_pins:
            sp.time = t


class OptimalTrajectory:
    def __init__(self, order, ndims, waypoints, min_derivative_order=4, continuity_order=2, constraint_check_dt=0.05):
        self.ndims = ndims
        self.order = order
        self.waypoints = waypoints
        self.solved = False
        self.splines = []
        self.min_derivative_order = min_derivative_order
        self.continuity_order = continuity_order
        self.num_segs = len(waypoints) - 1
        self.constraint_check_dt = constraint_check_dt
        assert self.num_segs >= 1

    def solve(self, aggressiveness=0.1, time_opt_order=2):
        self._aggro = aggressiveness
        self._time_opt_order = time_opt_order

        minperseg = 50.0/1000
        res = scipy.optimize.minimize(
            self._cost_fn,
            np.ones(self.num_segs),
            bounds=[(minperseg, np.inf) for i in range(self.num_segs)],
            options={'disp': False})
        x = res.x
        ts = np.hstack((np.array([0]), np.cumsum(x)))
        for i, wp in enumerate(self.waypoints):
            wp.set_time(ts[i])

        self.splines = self._gen_splines()
        self.solved = True

    def val(self, t, dim=None, order=0):
        if not self.solved:
            print("TRAJECTORY NOT SOLVED!!")
            return None

        if dim is None:
            return [s.val(order, t) for s in self.splines]

        return self.splines[dim].val(order, t)

    def end_time(self):
        return self.waypoints[-1].time

    def _cost_fn(self, x):
        # return sum([max(x, 0) for x in x]) + self._nl_constraints_fn(x).transpose() @ self.derivative_weights @ self._nl_constraints_fn(x)
        return self._aggro * sum(x) + self._compute_avg_cost_per_dim(x)

    def _compute_avg_cost_per_dim(self, x):
        ts = np.hstack((np.array([0]), np.cumsum(x)))
        for i, wp in enumerate(self.waypoints):
            wp.set_time(ts[i])

        splines = self._gen_splines()

        order = self.order
        num_segments = self.num_segs

        cw = order + 1  # constraint width
        x_dim = cw * (num_segments)  # num columns in the constraint matrix

        # Construct Hermitian matrix:
        H = np.zeros((x_dim, x_dim))
        for seg in range(0, num_segments):
            Q = self._compute_Q(order, self._time_opt_order, 0, x[seg])
            H[cw * seg:cw * (seg + 1), cw * seg:cw * (seg + 1)] = Q

        res = 0
        for spline in splines:
            try:
                c = spline._get_coeff_vector()
                res += c.dot(H.dot(c.transpose()))
            except:
                print('fuck')
                res += 10000
        return res / self.ndims / ts[-1]

    def _compute_Q(self, order, min_derivative_order, t1, t2):
        r = min_derivative_order
        n = order

        T = np.zeros((n - r) * 2 + 1)
        for i in range(0, len(T)):
            T[i] = t2 ** (i + 1) - t1 ** (i + 1)

        Q = np.zeros((n + 1, n + 1))

        for i in range(r, n + 1):
            for j in range(i, n + 1):
                k1 = i - r
                k2 = j - r
                k = k1 + k2 + 1
                Q[i, j] = self._dc(k1, k1 + r) * self._dc(k2, k2 + r) / k * T[k - 1]
                Q[j, i] = Q[i, j]
        return Q

    def _calc_tvec(self, t, polynomial_order, tvec_order):
        r = tvec_order
        n = polynomial_order
        tvec = np.zeros(n + 1)
        for i in range(r, n + 1):
            tvec[i] = self._dc(i - r, i) * t ** (i - r)
        return tvec

    def _dc(self, d, p):
        return factorial(p) / factorial(d)

    def _gen_splines(self):
        splines = [None] * self.ndims
        for i in range(self.ndims):
            pins = [wp.spline_pins[i] for wp in self.waypoints]
            splines[i] = OptimalSplineGen.compute_min_derivative_spline(self.order,
                                                                        self.min_derivative_order,
                                                                        self.continuity_order,
                                                                        pins)
        return splines

