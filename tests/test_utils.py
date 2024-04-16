import unittest
import torch
from utils import get_last_masked_index, get_random_masked_index


class Test(unittest.TestCase):

    def test_get_last_in_sequence_single_mask(self):
        mask = torch.tensor([[1, 1, 0, 0]])
        result = get_last_masked_index(mask)
        self.assertIsNotNone(result)
        self.assertEqual(result.item(), 1)

    def test_get_last_in_sequence_multiple_masks(self):
        mask = torch.tensor([[1, 1, 0, 0],
                             [1, 1, 0, 1]])
        result = get_last_masked_index(mask)
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, torch.tensor([1, 3])))

    def test_random_token(self):
        mask = torch.tensor([[1, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 1, 1, 0]])
        result = get_random_masked_index(mask)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3,))


if __name__ == "__main__":
    unittest.main()
