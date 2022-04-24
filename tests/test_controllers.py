from unittest import TestCase

import controllers

NAME = 'PID'
PID = (10, 0.1, 2)


class TestPIDController(TestCase):

    def setUp(self) -> None:
        self.controller = controllers.PIDController(PID)

    def test_set_control(self):
        pid = (5, 1, 9)
        self.controller.set_control(pid)
        self.assertEqual(5, self.controller.p_control)
        self.assertEqual(1, self.controller.i_control)
        self.assertEqual(9, self.controller.d_control)

    def test_set_control_failure(self):
        pid = (1, 4, 5, 6)
        with self.assertRaises(ValueError):
            self.controller.set_control(pid)

    def test_get_control(self):
        pid = self.controller.get_control()
        self.assertEqual(PID, pid)
        self.assertEqual(10, self.controller.p_control)
        self.assertEqual(0.1, self.controller.i_control)
        self.assertEqual(2, self.controller.d_control)

    def test_get_output_positive(self):
        setpoint, value = 100, 10
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(90, self.controller.i_value)
        self.assertEqual(90, self.controller.d_value)
        self.assertEqual(900 + 9 + 180, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(180, self.controller.i_value)
        self.assertEqual(90, self.controller.d_value)
        self.assertEqual(900 + 18 + 0, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(270, self.controller.i_value)
        self.assertEqual(90, self.controller.d_value)
        self.assertEqual(900 + 27 + 0, output)

    def test_get_output_negative(self):
        setpoint, value = -10, -1
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(-90 + -0.9 + -18, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(-90 + -1.8 + 0, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(-90 + -2.7 + 0, output)

    def test_reset(self):
        setpoint, value = 100, 10
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(90, self.controller.i_value)
        self.assertEqual(90, self.controller.d_value)
        self.controller.reset()
        self.assertEqual(0, self.controller.i_value)
        self.assertEqual(0, self.controller.d_value)


class TestLearningPIDController(TestCase):
    def setUp(self) -> None:
        self.controller = controllers.LearningPIDController(NAME, PID)

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(10.0, 0.1, 2.0)', text)

    def test_reward(self):
        self.controller.reward(10)
        self.assertEqual([10], self.controller.current_rewards)
        self.assertEqual([], self.controller.previous_rewards)

    def test_get_improvement_positive_not_zero(self):
        result = self.controller.get_improvement(10, 1)
        self.assertEqual(10, result)

        result = self.controller.get_improvement(10, 2)
        self.assertEqual(5, result)

        result = self.controller.get_improvement(10, 10)
        self.assertEqual(1, result)

        result = self.controller.get_improvement(10, 100)
        self.assertEqual(0.1, result)

        result = self.controller.get_improvement(0.1, 100)
        self.assertEqual(0.001, result)

    def test_get_improvement_positive_at_zero(self):
        result = self.controller.get_improvement(0, 0.1)
        self.assertLess(result, 0.5)

        result = self.controller.get_improvement(0, 1)
        self.assertLess(result, 0.334)

        result = self.controller.get_improvement(0, 10)
        self.assertLess(result, 0.1)

        result = self.controller.get_improvement(10, 0)
        self.assertGreater(result, 1)

        result = self.controller.get_improvement(1, 0)
        self.assertGreater(result, 1)

        result = self.controller.get_improvement(0.1, 0)
        self.assertGreater(result, 1)

        result = self.controller.get_improvement(0.01, 0)
        self.assertGreater(result, 1)

    def test_get_improvement_both_zero(self):
        result = self.controller.get_improvement(0, 0)
        self.assertGreater(result, 0.5)

    def test_get_improvement_both_equals(self):
        result = self.controller.get_improvement(-10, -10)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(-2, -2)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(-1.1, -1.1)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(-1, -1)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(-0.9, -0.9)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(-0.1, -0.1)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(0, 0)
        self.assertGreater(result, 0.5)

        result = self.controller.get_improvement(0.1, 0.1)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(0.9, 0.9)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(1, 1)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(1.1, 1.1)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(2, 2)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(10, 10)
        self.assertGreater(result, 0.5)
        result = self.controller.get_improvement(100, 100)
        self.assertGreater(result, 0.5)

    def test_get_improvement_negative_not_zero(self):
        result = self.controller.get_improvement(-10, -0.1)
        self.assertLess(result, 0.1)
        result = self.controller.get_improvement(-10, -1)
        self.assertLess(result, 0.1)
        result = self.controller.get_improvement(-10, -5)
        self.assertLess(result, 0.167)
        result = self.controller.get_improvement(-10, -9)
        self.assertLess(result, 0.5)

        result = self.controller.get_improvement(-10, -20)
        self.assertGreater(result, 1.0)
        result = self.controller.get_improvement(-1, -20)
        self.assertGreater(result, 1.0)
        result = self.controller.get_improvement(-0.1, -20)
        self.assertGreater(result, 1.0)
        result = self.controller.get_improvement(-0.1, -0.2)
        self.assertGreater(result, 1.0)

    def test_get_improvement_negative_at_zero(self):
        result = self.controller.get_improvement(-0.1, 0)
        self.assertLess(result, 0.5)
        result = self.controller.get_improvement(-1, 0)
        self.assertLess(result, 0.334)
        result = self.controller.get_improvement(-10, 0)
        self.assertLess(result, 0.1)

        result = self.controller.get_improvement(0, -10)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(0, -1)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(0, -0.1)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(0, -0.01)
        self.assertGreater(result, 1)

    def test_get_improvement_positive_negative(self):
        result = self.controller.get_improvement(10, -10)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(-10, 10)
        self.assertLess(result, 0.1)

        result = self.controller.get_improvement(1, -10)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(-10, 1)
        self.assertLess(result, 0.1)

        result = self.controller.get_improvement(1, -1)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(-1, 1)
        self.assertLess(result, 0.5)

        result = self.controller.get_improvement(10, -1)
        self.assertGreater(result, 1)
        result = self.controller.get_improvement(-1, 10)
        self.assertLess(result, 0.1)
