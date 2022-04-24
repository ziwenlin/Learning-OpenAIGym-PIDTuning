from unittest import TestCase

import controllers

NAME = 'PID'
PID = (10, 0.1, 2)


class Test_PID_Controller(TestCase):

    def setUp(self) -> None:
        self.controller = controllers.PIDController(PID)

    def test_set_control(self):
        pid = (5, 1, 9)
        self.controller.set_control(pid)
        self.assertEqual(self.controller.p_control, 5)
        self.assertEqual(self.controller.i_control, 1)
        self.assertEqual(self.controller.d_control, 9)

    def test_set_control_failure(self):
        pid = (1, 4, 5, 6)
        with self.assertRaises(ValueError):
            self.controller.set_control(pid)

    def test_get_control(self):
        pid = self.controller.get_control()
        self.assertEqual(PID, pid)
        self.assertEqual(self.controller.p_control, 10)
        self.assertEqual(self.controller.i_control, 0.1)
        self.assertEqual(self.controller.d_control, 2)

    def test_get_output_positive(self):
        setpoint, value = 100, 10
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(self.controller.i_value, 90)
        self.assertEqual(self.controller.d_value, 90)
        self.assertEqual(output, 900 + 9 + 180)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(self.controller.i_value, 180)
        self.assertEqual(self.controller.d_value, 90)
        self.assertEqual(output, 900 + 18 + 0)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(self.controller.i_value, 270)
        self.assertEqual(output, 900 + 27 + 0)

    def test_get_output_negative(self):
        setpoint, value = -10, -1
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(output, -90 + -0.9 + -18)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(output, -90 + -1.8 + 0)
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(output, -90 + -2.7 + 0)


    def test_reset(self):
        setpoint, value = 100, 10
        output = self.controller.get_output(value, setpoint)
        self.assertEqual(self.controller.i_value, 90)
        self.assertEqual(self.controller.d_value, 90)
        self.controller.reset()
        self.assertEqual(self.controller.i_value, 0)
        self.assertEqual(self.controller.d_value, 0)
