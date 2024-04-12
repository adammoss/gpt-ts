import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from tokenizer import LCTokenizer


class TestLCTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = LCTokenizer(
            min_flux=0,
            max_flux=1,
            num_bins=5,
            max_delta_time=10,
            num_time_bins=10,
            bands=[1, 2, 3],
            pad_token=True,
            min_delta_time=0,
            band_column='band',
            time_column='time',
            parameter_column='flux',
            parameter_error_column='flux_err',
            min_sn=0,
            window_size=10
        )

    def test_flux_token(self):
        flux = 0.3
        result = self.tokenizer.flux_token(flux)
        self.assertEqual(result, 1)

    def test_time_token(self):
        delta_time = 5
        result = self.tokenizer.time_token(delta_time)
        self.assertEqual(result, 5)

    def test_encode(self):
        df = pd.DataFrame({
            'object_id': [1, 1, 2, 2],
            'time': [0, 1, 0, 1],
            'band': [1, 2, 1, 2],
            'flux': [0.1, 0.2, 0.3, 0.4],
            'flux_err': [0.01, 0.02, 0.03, 0.04]
        })

        result = self.tokenizer.encode(df)
        self.assertIsInstance(result, dict)
        self.assertIn(1, result.keys())
        self.assertIsInstance(result[1], list)

    def test_decode(self):
        tokens = [3, 7, 12]
        result = self.tokenizer.decode(tokens)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
