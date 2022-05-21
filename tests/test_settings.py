from unittest import TestCase

import settings


class Test(TestCase):

    def test_get_dict_not_includes(self):
        with self.assertRaises(KeyError):
            a = settings.get_dict()['EPISODE.MULTIPLIER']

    def test_get_dict_includes_changes(self):
        settings.EPISODE.SHOW = 1
        self.assertEqual(1, settings.get_dict()['EPISODE.SHOW'])
        settings.EPISODE.SHOW = 10
        self.assertEqual(10, settings.get_dict()['EPISODE.SHOW'])
        settings.EPSILON.DISCOUNT = 1
        self.assertEqual(1, settings.get_dict()['EPSILON.DISCOUNT'])
        settings.EPSILON.DISCOUNT = 10
        self.assertEqual(10, settings.get_dict()['EPSILON.DISCOUNT'])

    def test_set_dict_not_includes(self):
        info = settings.get_dict()
        info['EPISODE.MULTIPLIER'] = 1000
        settings.EPISODE.MULTIPLIER = 1
        self.assertEqual(1, settings.EPISODE.MULTIPLIER)
        settings.set_dict(info)
        self.assertEqual(1, settings.EPISODE.MULTIPLIER)
        info['EPSILON.DECAY_RATE'] = 100
        settings.EPSILON.DECAY_RATE = 1
        self.assertEqual(1, settings.EPSILON.DECAY_RATE)
        settings.set_dict(info)
        self.assertEqual(1, settings.EPSILON.DECAY_RATE)

    def test_set_dict_includes_changes(self):
        info = settings.get_dict()
        info['EPISODE.SHOW'] = 1000
        settings.EPISODE.SHOW = 1
        self.assertEqual(1, settings.EPISODE.SHOW)
        settings.set_dict(info)
        self.assertEqual(1000, settings.EPISODE.SHOW)
        info['EPSILON.VALUE'] = 100
        settings.EPSILON.VALUE = 1
        self.assertEqual(1, settings.EPSILON.VALUE)
        settings.set_dict(info)
        self.assertEqual(100, settings.EPSILON.VALUE)
