from unittest import TestCase

from src import controllers
from src import genetics


class TestModelManager(TestCase):
    def setUp(self) -> None:
        self.controller = genetics.ModelManager()
        self.assertEqual(0, len(self.controller.models))
        self.controller.add_model(
            controllers.PIDModel((0, 2, 0)))
        self.controller.add_model(
            controllers.PIDModel((3, 0, 1)))
        self.controller.add_model(
            controllers.PIDModel((0, 1, 4)))

    def test_add_model_is_adding(self):
        self.assertEqual(3, len(self.controller.models))
        self.assertEqual(
            (0, 2, 0), self.controller.models[0].get_model())
        self.assertEqual(
            (3, 0, 1), self.controller.models[1].get_model())
        self.assertEqual(
            (0, 1, 4), self.controller.models[2].get_model())

    def test_set_models_changes_models(self):
        self.controller.set_models([
            (0, 2, 1), (3, 5, 1), (3, 1, 4)
        ])
        self.assertEqual([
            (0, 2, 1), (3, 5, 1), (3, 1, 4)
        ], self.controller.get_models())

    def test_set_models_value_error(self):
        with self.assertRaises(ValueError):
            self.controller.set_models([
                (1,), 2, 3, 4
            ])
        self.assertEqual([
            (0, 2, 0), (3, 0, 1), (0, 1, 4)
        ], self.controller.get_models())

    def test_get_models_returns_list_tuple(self):
        self.assertEqual([
            (0, 2, 0), (3, 0, 1), (0, 1, 4)
        ], self.controller.get_models())

    def test_get_string(self):
        text = self.controller.get_string()
        self.assertEqual(
            '[(0, 2, 0), (3, 0, 1), (0, 1, 4), ]', text
        )

    # def test_explore_replace(self):
    #     self.controller.models[0]
    #     self.controller.explore_replace([(0, 0, 0), (1, 1, 1), (2, 2, 2), ])
    #     self.assertEqual((0, 0, 0), self.controller.controls[0])
    #     self.assertEqual((1, 1, 1), self.controller.controls[1])
    #     self.assertEqual((2, 2, 2), self.controller.controls[2])
    #     self.assertEqual(
    #         (0, 0, 0), self.controller.models[0].get_model())
    #     self.assertEqual(
    #         (1, 1, 1), self.controller.models[1].get_model())
    #     self.assertEqual(
    #         (2, 2, 2), self.controller.models[2].get_model())

    # def test_explore_replace_lifetime_reset(self):
    #     self.controller.lifetime = 10
    #     self.controller.explore_replace([(0, 0, 0), (1, 1, 1), (2, 2, 2), ])
    #     self.assertEqual(0, self.controller.lifetime)

    # def test_explore_breed(self):
    #     self.controller.explore_breed([((1, 3, 4), 2)])
    #     self.assertEqual((1, 3, 4), self.controller.controls[2])
    #     self.assertEqual(
    #         (1, 3, 4), self.controller.models[2].get_model())

    # def test_reward(self):
    #     self.controller.reward(10)
    #     self.controller.reward(5)
    #     self.controller.reward(2)
    #     self.assertEqual([10, 5, 2], self.controller.current_rewards)
    #     self.assertEqual([], self.controller.previous_rewards)
    #
    # def test_reflect(self):
    #     self.assertEqual(0, self.controller.lifetime)
    #     self.controller.reward(10)
    #     self.controller.reflect()
    #     self.assertEqual([], self.controller.current_rewards)
    #     self.assertEqual([10], self.controller.previous_rewards)
    #     self.assertEqual(1, self.controller.lifetime)


class TestGeneticEvolutionController(TestCase):
    # Todo GeneticController BaseTest
    # LearningController.BaseTest is forbidden here.
    def setUp(self) -> None:
        self.controller = genetics.GeneticEvolutionController()
        self.controller.add_controller(
            controllers.PIDModel((0, 2, 0)))
        self.controller.add_controller(
            controllers.PIDModel((3, 0, 1)))
        self.controller.add_controller(
            controllers.PIDModel((0, 1, 4)))

    def test_reflect_reward_get_saved(self):
        self.controller.current_rewards = [0, 2, 4, 6]
        # self.controller.reflect()
        # self.assertEqual([], self.controller.current_rewards)
        # self.assertEqual([0, 2, 4, 6], self.controller.previous_rewards[0])
        # self.controller.current_rewards = [3, 2, 4, 6]
        # self.controller.reflect()
        # self.assertEqual([0, 2, 4, 6], self.controller.previous_rewards[0])
        # self.assertEqual([3, 2, 4, 6], self.controller.previous_rewards[1])
