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

    def test_set_dict_not_includes(self):
        info = settings.get_dict()
        info['EPISODE.MULTIPLIER'] = 1000
        settings.EPISODE.MULTIPLIER = 1
        self.assertEqual(1, settings.EPISODE.MULTIPLIER)
        settings.set_dict(info)
        self.assertEqual(1, settings.EPISODE.MULTIPLIER)

    def test_set_dict_includes_changes(self):
        info = settings.get_dict()
        info['EPISODE.SHOW'] = 1000
        settings.EPISODE.SHOW = 1
        self.assertEqual(1, settings.EPISODE.SHOW)
        settings.set_dict(info)
        self.assertEqual(1000, settings.EPISODE.SHOW)
