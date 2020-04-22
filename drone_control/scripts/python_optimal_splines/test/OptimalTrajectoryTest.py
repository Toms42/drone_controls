import unittest
import OptimalSplineGen
from OptimalTrajectory import TrajectoryWaypoint, OptimalTrajectory
import numpy as np


class TestOptimalSplineGen(unittest.TestCase):
    # def test_waypoints(self):
    #     tw = TrajectoryWaypoint(3)
    #     tw.add_hard_constraints(0, (1, 2, 3))
    #     self.assertEqual(len(tw.spline_pins), 3)
    #     self.assertEqual(len(tw.spline_pins[0]), 1)
    #     self.assertEqual(tw.spline_pins[0][0].hard_constraints[0], (0, 1))
    #     self.assertEqual(tw.spline_pins[1][0].hard_constraints[0], (0, 2))
    #     self.assertEqual(tw.spline_pins[2][0].hard_constraints[0], (0, 3))

    def test_gen_splines(self):
        tw1 = TrajectoryWaypoint((1, 2, 3))
        tw1.add_hard_constraints(1, (0, 0, 0))
        tw2 = TrajectoryWaypoint((2, 0, 5))
        tw3 = TrajectoryWaypoint((1, 4, 7))
        tw3.add_hard_constraints(1, (0, 0, 0))

        tw1.set_time(0)
        tw2.set_time(2)
        tw3.set_time(3)

        wpts = [tw1, tw2, tw3]
        ot = OptimalTrajectory(5, 3, wpts)
        splines = ot._gen_splines()
        self.assertAlmostEqual(splines[0].val(0, 0), 1, places=2)
        self.assertAlmostEqual(splines[0].val(0, 2), 2, places=2)
        self.assertAlmostEqual(splines[0].val(0, 3), 1, places=2)
        self.assertAlmostEqual(splines[1].val(0, 0), 2, places=2)
        self.assertAlmostEqual(splines[1].val(0, 2), 0, places=2)
        self.assertAlmostEqual(splines[1].val(0, 3), 4, places=2)
        self.assertAlmostEqual(splines[2].val(0, 0), 3, places=2)
        self.assertAlmostEqual(splines[2].val(0, 2), 5, places=2)
        self.assertAlmostEqual(splines[2].val(0, 3), 7, places=2)
        self.assertAlmostEqual(splines[0].val(1, 0), 0, places=2)
        self.assertAlmostEqual(splines[0].val(1, 3), 0, places=2)
        self.assertAlmostEqual(splines[2].val(1, 0), 0, places=2)
        self.assertAlmostEqual(splines[2].val(1, 3), 0, places=2)
    #
    # def test_nl_constraints_fn(self):
    #     tw1 = TrajectoryWaypoint((1, 2, 3))
    #     tw1.add_hard_constraints(1, (0, 0, 0))
    #     tw2 = TrajectoryWaypoint((2, 0, 5))
    #     tw3 = TrajectoryWaypoint((1, 4, 7))
    #     tw3.add_hard_constraints(1, (0, 0, 0))
    #
    #     wpts = [tw1, tw2, tw3]
    #     ot = OptimalTrajectory(5, 3, wpts)
    #     v = ot._nl_constraints_fn([10, 10])
    #     v2 = ot._nl_constraints_fn([5, 5])
    #     print(v, v2)

    def test_solve(self):
        tw1 = TrajectoryWaypoint((1, 2, 3))
        tw1.add_hard_constraints(1, (0, 0, 0))
        tw2 = TrajectoryWaypoint((2, 0, 4))
        tw3 = TrajectoryWaypoint((2, 4, 3))
        tw4 = TrajectoryWaypoint((4, 4, 5))
        tw5 = TrajectoryWaypoint((40, 23, 24))
        tw6 = TrajectoryWaypoint((4, 2, 5))
        tw7 = TrajectoryWaypoint((2, 6, 3))
        tw7.add_hard_constraints(1, (0, 0, 0))

        wpts = [tw1, tw2, tw3, tw4, tw5, tw6, tw7]
        ot = OptimalTrajectory(5, 3, wpts)
        ot.solve()



if __name__ == '__main__':
    unittest.main()
