import unittest
from unittest.mock import patch, Mock
import pandas as pd

from gaussian_process import fit_2d_gp


class Fit2dGpTest(unittest.TestCase):
    def setUp(self):
        # Preparing a minimal setup for test data

        self.sample_data = pd.DataFrame({
            'mjd': [59750.4229, 59750.4306, 59750.4383, 59750.445, 59752.407,
                    59752.4147, 59752.4224, 59752.4334, 59752.4435, 59767.2968,
                    59767.3045, 59767.3122, 59767.3233, 59767.3343, 59770.2179,
                    59770.2256, 59770.2334, 59770.2445, 59770.2557, 59779.3188],
            'passband': [2, 1, 3, 4, 2, 1, 3, 4, 5, 2, 1, 3, 4, 5, 2, 1, 3, 4, 5, 2],
            'flux': [-544.810303, -816.434326, -471.385529, -388.984985,
                     -681.858887, -1061.457031, -524.95459, -393.480225,
                     -355.88678, -548.01355, -815.188599, -475.516052,
                     -405.663818, -421.199066, -554.903198, -820.042786,
                     -477.00473, -400.270386, -415.286896, -630.523682],
            'flux_err': [3.622952, 5.55337, 3.801213, 11.395031, 4.041204, 6.472994,
                         3.552751, 3.599346, 10.421921, 3.462291, 5.293019, 3.340643,
                         3.496113, 6.377517, 3.927843, 5.875329, 3.736262, 3.834955,
                         7.435979, 4.333287]
        })

        self.pb_wavelengths = {
            0: 3685.0,
            1: 4802.0,
            2: 6231.0,
            3: 7542.0,
            4: 8690.0,
            5: 9736.0,
        }

    def test_fit_2d_gp(self):
        # Testing the function

        result = fit_2d_gp(self.sample_data, self.pb_wavelengths)

        # Adding assertions to check if function gives expected results
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertEqual(len(result[0]), len(self.sample_data))

    @patch("george.GP")
    def test_fit_2d_gp_with_mock(self, mock_gp):
        # Testing with a mocked george.GP to simulate failed operation

        mock_gp.return_value.log_likelihood.side_effect = ValueError("Fail operation")

        with self.assertRaises(ValueError):
            fit_2d_gp(self.sample_data, self.pb_wavelengths)


if __name__ == '__main__':
    unittest.main()
