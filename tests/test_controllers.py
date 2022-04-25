from unittest import TestCase

import controllers

NAME = 'TEST'
PID = (10, 0.1, 2)
NODE = (0.5, 0.1, 2, 0)


class InOutController:
    class BaseTest(TestCase):
        def setUp(self) -> None:
            self.controller = controllers.InOutController()

        def test_output_reset(self):
            self.assertEqual(0, self.controller.output)
            self.controller.output = 10
            self.controller.reset()
            self.assertEqual(0, self.controller.output)


class TestPIDController(InOutController.BaseTest):
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


class TestNodeController(InOutController.BaseTest):
    def setUp(self) -> None:
        self.controller = controllers.NodeController(NODE)

    def test_set_control(self):
        self.controller.set_control((0, 1, 2, 3))
        self.assertEqual((0, 1, 2, 3), self.controller.control)

    def test_get_control(self):
        control = self.controller.get_control()
        self.assertEqual((0.5, 0.1, 2, 0), control)

    def test_get_output(self):
        output = self.controller.get_output((0, 10, -2, 1), 0)
        self.assertEqual(0 + 1 + -4 + 0, output)


class LearningInOutController:
    class BaseTest(TestCase):
        def setUp(self) -> None:
            self.controller = controllers.LearningInOutController()

        def test_reward_being_added_to_current_reward(self):
            for i in [1, 4, 6, 7, 3]:  # 21
                self.controller.reward(i)
            rewards = self.controller.current_rewards
            self.assertEqual([1, 4, 6, 7, 3], rewards)
            rewards = self.controller.previous_rewards
            self.assertEqual([], rewards)

        def test_reflect_processsed_current_and_previous_rewards(self):
            for i in [1, 4, 6, 7, 3]:  # 21
                self.controller.reward(i)
            self.controller.reflect()
            rewards = self.controller.current_rewards
            self.assertEqual([], rewards)
            rewards = self.controller.previous_rewards
            self.assertEqual([1, 4, 6, 7, 3], rewards)

        def test_reward_being_added_to_current_reward_after_reflect(self):
            for i in [1, 4, 6, 7, 3]:  # 21
                self.controller.reward(i)
            self.controller.reflect()
            for i in [5, 7, 9, 3, 1]:  # 25
                self.controller.reward(i)
            rewards = self.controller.current_rewards
            self.assertEqual([5, 7, 9, 3, 1], rewards)
            rewards = self.controller.previous_rewards
            self.assertEqual([1, 4, 6, 7, 3], rewards)

        def test_reflect_lower_reward_at_second_reflect(self):
            for i in [5, 7, 9, 3, 1]:  # 25
                self.controller.reward(i)
            self.controller.reflect()
            for i in [1, 4, 6, 7, 3]:  # 21
                self.controller.reward(i)
            self.controller.reflect()

            rewards = self.controller.current_rewards
            self.assertEqual([], rewards)
            rewards = self.controller.previous_rewards
            self.assertEqual([5, 7, 9, 3, 1], rewards)

        def test_reflect_higher_reward_at_second_reflect(self):
            for i in [1, 4, 6, 7, 3]:  # 21
                self.controller.reward(i)
            self.controller.reflect()
            for i in [5, 7, 9, 3, 1]:  # 25
                self.controller.reward(i)
            self.controller.reflect()

            rewards = self.controller.current_rewards
            self.assertEqual([], rewards)
            rewards = self.controller.previous_rewards
            self.assertEqual([5, 7, 9, 3, 1], rewards)


class TestLearningInOutController(LearningInOutController.BaseTest):
    def setUp(self) -> None:
        # Using PID because the following tests need an implemented controller
        self.controller = controllers.LearningPIDController()

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(0.0, 0.0, 0.0)', text)

    def test_reflect_previous_control_improved_empty_rewards(self):
        self.controller.current_control = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_control)

        self.controller.reflect()
        self.assertEqual((1, 2, 3), self.controller.current_control)
        self.assertEqual((1, 2, 3), self.controller.previous_control)

    def test_reflect_previous_control_should_not_change(self):
        self.controller.previous_rewards = [10]
        self.controller.current_rewards = [-1]
        self.controller.current_control = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_control)

        self.controller.reflect()
        self.assertEqual((0, 0, 0), self.controller.current_control)
        self.assertEqual((0, 0, 0), self.controller.previous_control)

    def test_reflect_previous_control_should_change(self):
        self.controller.previous_rewards = [10]
        self.controller.current_rewards = [100]
        self.controller.current_control = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_control)

        self.controller.reflect()
        self.assertEqual((1, 2, 3), self.controller.current_control)
        self.assertEqual((1, 2, 3), self.controller.previous_control)


class TestLearningNodeController(LearningInOutController.BaseTest):
    def setUp(self) -> None:
        self.controller = controllers.LearningNodeController(NAME, NODE)

    def test_init_none_preset(self):
        with self.assertRaises(ValueError):
            controllers.LearningNodeController()

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(0.5, 0.1, 2.0, 0.0)', text)


class TestLearningPIDController(LearningInOutController.BaseTest):
    def setUp(self) -> None:
        self.controller = controllers.LearningPIDController(NAME, PID)

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(10.0, 0.1, 2.0)', text)


class Test_get_improvement(TestCase):
    def test_get_improvement_positive_not_zero(self):
        result = controllers.get_improvement(10, 1)
        self.assertEqual(10, result)

        result = controllers.get_improvement(10, 2)
        self.assertEqual(5, result)

        result = controllers.get_improvement(10, 10)
        self.assertEqual(1, result)

        result = controllers.get_improvement(10, 100)
        self.assertEqual(0.1, result)

        result = controllers.get_improvement(0.1, 100)
        self.assertEqual(0.001, result)

    def test_get_improvement_positive_at_zero(self):
        result = controllers.get_improvement(0, 0.01)
        self.assertLess(result, 0.5)
        result = controllers.get_improvement(0, 0.1)
        self.assertLess(result, 0.5)
        result = controllers.get_improvement(0, 1)
        self.assertLess(result, 0.334)
        result = controllers.get_improvement(0, 10)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement(10, 0)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(1, 0)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(0.1, 0)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(0.01, 0)
        self.assertGreater(result, 1)

    def test_get_improvement_both_zero(self):
        result = controllers.get_improvement(0, 0)
        self.assertGreater(result, 0.5)

    def test_get_improvement_both_equals(self):
        result = controllers.get_improvement(-10, -10)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(-2, -2)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(-1.1, -1.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(-1, -1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(-0.9, -0.9)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(-0.1, -0.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(0, 0)
        self.assertGreater(result, 0.5)

        result = controllers.get_improvement(0.1, 0.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(0.9, 0.9)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(1, 1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(1.1, 1.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(2, 2)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(10, 10)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement(100, 100)
        self.assertGreater(result, 0.5)

    def test_get_improvement_negative_not_zero(self):
        result = controllers.get_improvement(-10, -0.1)
        self.assertLess(result, 0.1)
        result = controllers.get_improvement(-10, -1)
        self.assertLess(result, 0.1)
        result = controllers.get_improvement(-10, -5)
        self.assertLess(result, 0.167)
        result = controllers.get_improvement(-10, -9)
        self.assertLess(result, 0.5)

        result = controllers.get_improvement(-10, -20)
        self.assertGreater(result, 1.0)
        result = controllers.get_improvement(-1, -20)
        self.assertGreater(result, 1.0)
        result = controllers.get_improvement(-0.1, -20)
        self.assertGreater(result, 1.0)
        result = controllers.get_improvement(-0.1, -0.2)
        self.assertGreater(result, 1.0)

    def test_get_improvement_negative_at_zero(self):
        result = controllers.get_improvement(-0.1, 0)
        self.assertLess(result, 0.5)
        result = controllers.get_improvement(-1, 0)
        self.assertLess(result, 0.334)
        result = controllers.get_improvement(-10, 0)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement(0, -10)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(0, -1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(0, -0.1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(0, -0.01)
        self.assertGreater(result, 1)

    def test_get_improvement_positive_negative(self):
        result = controllers.get_improvement(10, -10)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(-10, 10)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement(1, -10)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(-10, 1)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement(1, -1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(-1, 1)
        self.assertLess(result, 0.5)

        result = controllers.get_improvement(10, -1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement(-1, 10)
        self.assertLess(result, 0.1)


class TestInOutController(TestCase):
    def setUp(self) -> None:
        self.controller = controllers.InOutController()

    def test_set_control(self):
        with self.assertRaises(NotImplementedError):
            self.controller.set_control(object)

    def test_get_control(self):
        with self.assertRaises(NotImplementedError):
            self.controller.get_control()

    def test_get_output(self):
        with self.assertRaises(NotImplementedError):
            self.controller.get_output(object, object)

    def test_reset(self):
        with self.assertRaises(NotImplementedError):
            self.controller.reset()
