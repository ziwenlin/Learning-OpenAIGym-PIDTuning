from unittest import TestCase, mock

import controllers

NAME = 'TEST'
PID = (10, 0.1, 2)
NODE = (0.5, 0.1, 2, 0)


class BaseTest:
    class InOutController(TestCase):
        def setUp(self) -> None:
            self.controller = controllers.InOutController()

        def test_reset_output_is_implemented(self):
            self.assertEqual(0, self.controller.output)
            self.controller.output = 10
            self.controller.reset()
            self.assertEqual(0, self.controller.output)

        def test_set_control_is_implemented(self):
            # This might be implemented in the future
            pass

        def test_get_output_is_implemented(self):
            output = self.controller.get_output((10,), 0)
            self.assertNotEqual(0, output)
            self.assertEqual(output, self.controller.output)

        def test_get_control_is_implemented(self):
            control = self.controller.get_control()
            self.assertEqual(tuple, type(control))

    class ImprovingController(TestCase):
        def setUp(self) -> None:
            self.controller = controllers.ImprovingController()

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
            with mock.patch('numpy.random.rand', lambda: 0.9):
                for i in [5, 7, 9, 3, 1]:  # 25, 1, 9
                    self.controller.reward(i)
                self.controller.reflect()
                for i in [1, 4, 6, 7, 3]:  # 21, 1, 7
                    self.controller.reward(i)
                self.controller.reflect()

            rewards = self.controller.current_rewards
            self.assertEqual([], rewards)
            rewards = self.controller.previous_rewards
            self.assertEqual([5, 7, 9, 3, 1], rewards)

        def test_reflect_higher_reward_at_second_reflect(self):
            with mock.patch('numpy.random.rand', lambda: 0.9):
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

    class RotatingController(TestCase):
        def setUp(self) -> None:
            self.controller = controllers.RotatingController()

        # Todo add more tests here
        def test_next_controller_when_empty(self):
            if len(self.controller.controllers) == 0:
                with self.assertRaises(IndexError):
                    self.controller.next_controller()

        def test_select_controller_when_empty(self):
            if len(self.controller.controllers) == 0:
                with self.assertRaises(IndexError):
                    self.controller.select_controller(10)

        def test_add_controller_when_empty(self):
            controller = self.controller
            if len(self.controller.controllers) == 0:
                self.assertEqual(0, len(controller.controllers))
                controller.add_controller(controllers.ImprovingPIDController())
            self.assertEqual(1, len(controller.controllers))
            self.assertEqual(controller.selected,
                             controller.controllers[0])


# class TestInOutController(TestCase):
#     def setUp(self) -> None:
#         self.controller = controllers.InOutController()
#
#     def test_set_control(self):
#         with self.assertRaises(NotImplementedError):
#             self.controller.set_control(object)
#
#     def test_get_control(self):
#         with self.assertRaises(NotImplementedError):
#             self.controller.get_control()
#
#     def test_get_output(self):
#         with self.assertRaises(NotImplementedError):
#             self.controller.get_output(object, object)
#
#     def test_reset(self):
#         with self.assertRaises(NotImplementedError):
#             self.controller.reset()


class TestPIDController(BaseTest.InOutController):
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
        setpoint, value = 100, (10,)
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
        setpoint, value = -10, (-1,)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(-90 + -0.9 + -18, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(-90 + -1.8 + 0, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(-90 + -2.7 + 0, output)

    def test_reset(self):
        setpoint, value = 100, (10,)
        self.controller.get_output(value, setpoint)
        self.assertEqual(90, self.controller.i_value)
        self.assertEqual(90, self.controller.d_value)
        self.controller.reset()
        self.assertEqual(0, self.controller.i_value)
        self.assertEqual(0, self.controller.d_value)


class TestNodeController(BaseTest.InOutController):
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


class TestImprovingInOutController(BaseTest.ImprovingController):
    def setUp(self) -> None:
        # Using PID because the following tests need an implemented controller
        self.controller = controllers.ImprovingInOutController()

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
        with mock.patch('numpy.random.rand', lambda: 1.0 / 0.9):
            self.controller.reflect()
        self.assertEqual((0, 0, 0), self.controller.current_control)
        self.assertEqual((0, 0, 0), self.controller.previous_control)

    def test_reflect_previous_control_should_change(self):
        self.controller.previous_rewards = [10]
        self.controller.current_rewards = [100]
        self.controller.current_control = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_control)
        with mock.patch('numpy.random.rand', lambda: 1.0 / 0.9):
            self.controller.reflect()
        self.assertEqual((1, 2, 3), self.controller.current_control)
        self.assertEqual((1, 2, 3), self.controller.previous_control)


class TestLearningNodeController(BaseTest.ImprovingController):
    def setUp(self) -> None:
        self.controller = controllers.ImprovingNodeController(NAME, NODE)

    def test_init_none_preset(self):
        with self.assertRaises(ValueError):
            controllers.ImprovingNodeController()

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(0.5, 0.1, 2.0, 0.0)', text)


class TestLearningPIDController(BaseTest.ImprovingController):
    def setUp(self) -> None:
        self.controller = controllers.ImprovingPIDController(NAME, PID)

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(10.0, 0.1, 2.0)', text)


class TestRotatingImprovingController(BaseTest.ImprovingController,
                                      BaseTest.RotatingController):
    # Todo add more tests here
    def setUp(self) -> None:
        self.controller = controllers.RotatingImprovingController()
        self.controller.add_controller(controllers.ImprovingPIDController())

    def test_reflect_is_counting(self):
        self.assertEqual(0, self.controller.count)
        self.controller.reflect()
        self.assertEqual(1, self.controller.count)

    def test_reflect_trigger_next(self):
        self.controller.count = 8
        self.controller.reflect()
        self.assertFalse(self.controller.is_next)
        self.controller.reflect()
        self.assertEqual(10, self.controller.count)
        self.assertTrue(self.controller.is_next)
        self.controller.reflect()
        self.assertEqual(11, self.controller.count)
        self.assertTrue(self.controller.is_next)


class TestEnvironmentMonitor(TestCase):
    def setUp(self) -> None:
        self.monitor = controllers.EnvironmentMonitor()
        self.monitor.monitor({'reward': 10, 'difficulty': 0.3, 'episode': 1})
        self.monitor.monitor({'reward': 12, 'difficulty': 0.2, 'episode': 2})
        self.monitor.monitor({'reward': 8, 'difficulty': 0.4, 'episode': 3})

    def test_monitor_rewards(self):
        self.monitor.monitor({'reward': 10, 'difficulty': 0, 'episode': 1})
        self.monitor.monitor({'reward': 12, 'difficulty': 0, 'episode': 8})
        self.assertEqual(8, self.monitor.ep_buffer[4]['episode'])

    def test_process_clear_buffer(self):
        self.assertEqual(3, len(self.monitor.ep_buffer))
        self.monitor.process(3)
        self.assertEqual(0, len(self.monitor.ep_buffer))

    def test_process_nothing(self):
        monitor = controllers.EnvironmentMonitor()
        with self.assertRaises(IndexError):
            monitor.process(0)

    def test_process_result_ep(self):
        self.monitor.process(3)
        self.assertEqual(3, self.monitor.results[0]['ep'])


class TestEnvironmentMonitorResults(TestCase):
    def setUp(self) -> None:
        self.monitor = controllers.EnvironmentMonitor()
        self.monitor.monitor({'reward': 10, 'difficulty': 0.3, 'episode': 1})
        self.monitor.monitor({'reward': 12, 'difficulty': 0.4, 'episode': 2})
        self.monitor.monitor({'reward': 8, 'difficulty': 0.4, 'episode': 3})
        self.monitor.monitor({'reward': 10, 'difficulty': 0.3, 'episode': 4})
        self.monitor.monitor({'reward': 12, 'difficulty': 0.2, 'episode': 5})
        self.monitor.monitor({'reward': 8, 'difficulty': 0.4, 'episode': 6})
        self.monitor.monitor({'reward': 16, 'difficulty': 0.1, 'episode': 7})
        self.monitor.monitor({'reward': 20, 'difficulty': 0.1, 'episode': 8})
        self.monitor.monitor({'reward': 14, 'difficulty': 0.2, 'episode': 9})
        self.monitor.monitor({'reward': 24, 'difficulty': 0.0, 'episode': 10})

    def test_process_results_highest(self):
        self.monitor.process(10)
        result = self.monitor.results[0]
        self.assertEqual(24, result['reward']['highest'])
        self.assertEqual(0.0, result['difficulty']['highest'])
        self.assertEqual(10, result['episode']['highest'])

    def test_process_results_lowest(self):
        self.monitor.process(10)
        result = self.monitor.results[0]
        self.assertEqual(8, result['reward']['lowest'])
        self.assertEqual(0.4, result['difficulty']['lowest'])
        self.assertEqual(3, result['episode']['lowest'])

    def test_process_results_median(self):
        self.monitor.process(10)
        result = self.monitor.results[0]
        self.assertEqual(12, result['reward']['median'])
        self.assertEqual(0.4, result['difficulty']['median'])
        self.assertEqual(2, result['episode']['median'])

    def test_process_results_middle(self):
        self.monitor.process(10)
        result = self.monitor.results[0]
        self.assertEqual(16, result['reward']['middle'])
        self.assertEqual(0.1, result['difficulty']['middle'])
        self.assertEqual(7, result['episode']['middle'])

    def test_process_results_mean(self):
        self.monitor.process(10)
        result = self.monitor.results[0]
        self.assertEqual(14, result['reward']['average'])
        self.assertEqual(0.2, result['difficulty']['average'])
        self.assertEqual(9, result['episode']['average'])


class Test(TestCase):
    def test_get_improvement_positive_not_zero(self):
        result = controllers.get_improvement_gain(10, 1)
        self.assertEqual(10, result)

        result = controllers.get_improvement_gain(10, 2)
        self.assertEqual(5, result)

        result = controllers.get_improvement_gain(10, 10)
        self.assertEqual(1, result)

        result = controllers.get_improvement_gain(10, 100)
        self.assertEqual(0.1, result)

        result = controllers.get_improvement_gain(0.1, 100)
        self.assertEqual(0.001, result)

    def test_get_improvement_positive_at_zero(self):
        result = controllers.get_improvement_gain(0, 0.01)
        self.assertLess(result, 0.5)
        result = controllers.get_improvement_gain(0, 0.1)
        self.assertLess(result, 0.5)
        result = controllers.get_improvement_gain(0, 1)
        self.assertLess(result, 0.334)
        result = controllers.get_improvement_gain(0, 10)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement_gain(10, 0)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(1, 0)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(0.1, 0)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(0.01, 0)
        self.assertGreater(result, 1)

    def test_get_improvement_both_zero(self):
        result = controllers.get_improvement_gain(0, 0)
        self.assertGreater(result, 0.5)

    def test_get_improvement_both_equals(self):
        result = controllers.get_improvement_gain(-10, -10)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(-2, -2)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(-1.1, -1.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(-1, -1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(-0.9, -0.9)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(-0.1, -0.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(0, 0)
        self.assertGreater(result, 0.5)

        result = controllers.get_improvement_gain(0.1, 0.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(0.9, 0.9)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(1, 1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(1.1, 1.1)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(2, 2)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(10, 10)
        self.assertGreater(result, 0.5)
        result = controllers.get_improvement_gain(100, 100)
        self.assertGreater(result, 0.5)

    def test_get_improvement_negative_not_zero(self):
        result = controllers.get_improvement_gain(-10, -0.1)
        self.assertLess(result, 0.1)
        result = controllers.get_improvement_gain(-10, -1)
        self.assertLess(result, 0.1)
        result = controllers.get_improvement_gain(-10, -5)
        self.assertLess(result, 0.167)
        result = controllers.get_improvement_gain(-10, -9)
        self.assertLess(result, 0.5)

        result = controllers.get_improvement_gain(-10, -20)
        self.assertGreater(result, 1.0)
        result = controllers.get_improvement_gain(-1, -20)
        self.assertGreater(result, 1.0)
        result = controllers.get_improvement_gain(-0.1, -20)
        self.assertGreater(result, 1.0)
        result = controllers.get_improvement_gain(-0.1, -0.2)
        self.assertGreater(result, 1.0)

    def test_get_improvement_negative_at_zero(self):
        result = controllers.get_improvement_gain(-0.1, 0)
        self.assertLess(result, 0.5)
        result = controllers.get_improvement_gain(-1, 0)
        self.assertLess(result, 0.334)
        result = controllers.get_improvement_gain(-10, 0)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement_gain(0, -10)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(0, -1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(0, -0.1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(0, -0.01)
        self.assertGreater(result, 1)

    def test_get_improvement_positive_negative(self):
        result = controllers.get_improvement_gain(10, -10)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(-10, 10)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement_gain(1, -10)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(-10, 1)
        self.assertLess(result, 0.1)

        result = controllers.get_improvement_gain(1, -1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(-1, 1)
        self.assertLess(result, 0.5)

        result = controllers.get_improvement_gain(10, -1)
        self.assertGreater(result, 1)
        result = controllers.get_improvement_gain(-1, 10)
        self.assertLess(result, 0.1)

    def test_get_control_mutated_returns_tuple(self):
        result = controllers.get_control_mutated((0, 0), (0, 0))
        self.assertEqual(tuple, type(result))

    def test_get_control_mutated_improvement_is_different(self):
        result = controllers.get_control_mutated((0, 0), (0, 0))
        if result[0] == 0:
            self.assertNotEqual(0.0, result[1])
        else:
            self.assertNotEqual(0, result[0])
            self.assertEqual(0, result[1])
