from unittest import TestCase, mock

import controllers
import mutations

NAME = 'TEST'
PID = (10, 0.1, 2)
NODE = (0.5, 0.1, 2, 0)


class BaseTest:
    class InOutController(TestCase):
        def setUp(self) -> None:
            self.controller = controllers.InOutModel()

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
            control = self.controller.get_model()
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
            self.controller = controllers.LearningControllerManager()

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
                controller.add_controller(controllers.ImprovingPIDModel())
            self.assertEqual(1, len(controller.controllers))
            self.assertEqual(controller.selected,
                             controller.controllers[0])


# class TestInOutController(TestCase):
#     def setUp(self) -> None:
#         self.controller = controllers.InOutController()
#
#     def test_set_control(self):
#         with self.assertRaises(NotImplementedError):
#             self.controller.set_model(object)
#
#     def test_get_control(self):
#         with self.assertRaises(NotImplementedError):
#             self.controller.get_model()
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
        self.controller = controllers.PIDModel(PID)

    def test_set_control(self):
        pid = (5, 1, 9)
        self.controller.set_model(pid)
        self.assertEqual(5, self.controller.model_p)
        self.assertEqual(1, self.controller.model_i)
        self.assertEqual(9, self.controller.model_d)

    def test_set_control_failure(self):
        pid = (1, 4, 5, 6)
        with self.assertRaises(ValueError):
            self.controller.set_model(pid)

    def test_get_control(self):
        pid = self.controller.get_model()
        self.assertEqual(PID, pid)
        self.assertEqual(10, self.controller.model_p)
        self.assertEqual(0.1, self.controller.model_i)
        self.assertEqual(2, self.controller.model_d)

    def test_get_output_positive(self):
        setpoint, value = 100, (10,)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(90, self.controller.value_i)
        self.assertEqual(90, self.controller.value_d)
        self.assertEqual(900 + 9 + 180, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(180, self.controller.value_i)
        self.assertEqual(90, self.controller.value_d)
        self.assertEqual(900 + 18 + 0, output)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(270, self.controller.value_i)
        self.assertEqual(90, self.controller.value_d)
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
        self.assertEqual(90, self.controller.value_i)
        self.assertEqual(90, self.controller.value_d)
        self.controller.reset()
        self.assertEqual(0, self.controller.value_i)
        self.assertEqual(0, self.controller.value_d)


class TestNodeController(BaseTest.InOutController):
    def setUp(self) -> None:
        self.controller = controllers.NodeModel(NODE)

    def test_set_control(self):
        self.controller.set_model((0, 1, 2, 3))
        self.assertEqual((0, 1, 2, 3), self.controller.control)

    def test_get_control(self):
        control = self.controller.get_model()
        self.assertEqual((0.5, 0.1, 2, 0), control)

    def test_get_output(self):
        output = self.controller.get_output((0, 10, -2, 1), 0)
        self.assertEqual(0 + 1 + -4 + 0, output)


class TestImprovingInOutController(BaseTest.ImprovingController):
    def setUp(self) -> None:
        # Using PID because the following tests need an implemented controller
        self.controller = controllers.ImprovingNodeModel(preset=(0, 0, 0))

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(0, 0, 0)', text)

    def test_reflect_previous_control_improved_empty_rewards(self):
        self.controller.current_model = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_model)

        self.controller.reflect()
        self.assertEqual((1, 2, 3), self.controller.current_model)
        self.assertEqual((1, 2, 3), self.controller.previous_model)

    def test_reflect_previous_control_should_not_change(self):
        self.controller.previous_rewards = [10]
        self.controller.current_rewards = [-1]
        self.controller.current_model = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_model)
        with mock.patch('numpy.random.rand', lambda: 1.0 / 0.9):
            self.controller.reflect()
        self.assertEqual((0, 0, 0), self.controller.current_model)
        self.assertEqual((0, 0, 0), self.controller.previous_model)

    def test_reflect_previous_control_should_change(self):
        self.controller.previous_rewards = [10]
        self.controller.current_rewards = [100]
        self.controller.current_model = (1, 2, 3)
        self.assertEqual((0, 0, 0), self.controller.previous_model)
        with mock.patch('numpy.random.rand', lambda: 1.0 / 0.9):
            self.controller.reflect()
        self.assertEqual((1, 2, 3), self.controller.current_model)
        self.assertEqual((1, 2, 3), self.controller.previous_model)


class TestLearningNodeController(BaseTest.ImprovingController):
    def setUp(self) -> None:
        self.controller = controllers.ImprovingNodeModel(NAME, NODE)

    def test_init_none_preset(self):
        with self.assertRaises(ValueError):
            controllers.ImprovingNodeModel()

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(0.5, 0.1, 2, 0)', text)


class TestLearningPIDController(BaseTest.ImprovingController):
    def setUp(self) -> None:
        self.controller = controllers.ImprovingPIDModel(NAME, PID)

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual('(10, 0.1, 2)', text)


class TestRotatingImprovingController(BaseTest.ImprovingController,
                                      BaseTest.RotatingController):
    # Todo add more tests here
    def setUp(self) -> None:
        self.controller = controllers.ImprovingControllerManager()
        self.controller.add_controller(controllers.ImprovingPIDModel())

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
        self.assertEqual(8, self.monitor.buffer[4]['episode'])

    def test_process_clear_buffer(self):
        self.assertEqual(3, len(self.monitor.buffer))
        self.monitor.process(3)
        self.assertEqual(0, len(self.monitor.buffer))

    def test_process_nothing(self):
        monitor = controllers.EnvironmentMonitor()
        with self.assertRaises(IndexError):
            monitor.process(0)

    def test_process_result_ep(self):
        self.monitor.process(3)
        result = self.monitor.results[0]
        self.assertEqual(3, result['division'])

    def test_process_result_episode_is_int(self):
        self.monitor.process(3)
        result = self.monitor.results[0]
        self.assertIsInstance(result['episode']['average'], int)
        other_info = [[c] + list(item.values())
                      for c, item in result.items()
                      if type(item) is dict]
        self.assertIsInstance(other_info[2][2], int)

    def test_get_log_output(self):
        self.monitor.process(3)
        test = """
|   division |   highest |   average |   lowest |   median |   middle |   epsilon |   multiplier |
|------------|-----------|-----------|----------|----------|----------|-----------|--------------|
|          3 |        12 |        10 |        8 |       10 |       10 |       0.9 |           10 |

| category   |   highest |   average |   lowest |   median |   middle |
|------------|-----------|-----------|----------|----------|----------|
| reward     |      12   |      10   |      8   |     10   |     10   |
| difficulty |       0.2 |       0.3 |      0.4 |      0.3 |      0.3 |
| episode    |       2   |       1   |      3   |      1   |      1   |
"""
        compare = self.monitor.get_log()
        self.assertEqual(test, compare)


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


class TestMutations(TestCase):
    def test_get_control_mutated_returns_tuple(self):
        result = mutations.mutate_io_model((0, 0), (0, 0))
        self.assertEqual(tuple, type(result))

    def test_mutate_io_model_is_different_on_randomness(self):
        with mock.patch('numpy.random.rand', lambda: 0.0):
            result = mutations.mutate_io_model((0, 1), (0, 0))
        self.assertNotEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_mutate_io_model_is_different_on_randomness_(self):
        with mock.patch('numpy.random.rand', lambda: 0.0):
            result = mutations.mutate_io_model((1, 0), (0, 0))
        self.assertNotEqual(0, result[1])
        self.assertEqual(1, result[0])

    def test_mutate_io_model_made_change_with_improvement(self):
        with mock.patch('numpy.random.rand', lambda: 1.0):
            result = mutations.mutate_io_model((1, 0), (0, 0))
        self.assertEqual(1.8, result[0])
        self.assertEqual(0, result[1])

    def test_mutate_io_mode_made_change_without_improvement(self):
        with mock.patch('numpy.random.rand', lambda: 1.0):
            result = mutations.mutate_io_model((0, 0), (0, 0))
        self.assertNotEqual((0, 0), result)
