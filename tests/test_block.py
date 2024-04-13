import unittest
from models import gpt
import torch
from torch import nn


class TestCasualBlock(unittest.TestCase):

    def setUp(self):
        # Setup process before each test
        self.n_embd = 128
        self.n_head = 2
        self.n_positions = 1024
        self.position_embedding = None
        self.is_causal = True
        self.block = gpt.Block(self.n_embd, self.n_head, self.n_positions, position_embedding=self.position_embedding,
                               is_causal=self.is_causal)

    def test_block_init(self):
        # Testing for initialization
        self.assertIsInstance(self.block, gpt.Block)

    def test_module_type(self):
        # Checking type of transformer block
        self.assertIsInstance(self.block.sa, gpt.MultiHeadAttention)
        self.assertIsInstance(self.block.ffwd, gpt.FeedFoward)
        self.assertIsInstance(self.block.ln1, nn.LayerNorm)
        self.assertIsInstance(self.block.ln2, nn.LayerNorm)

    def test_forward(self):
        # Checking forward computation
        x = torch.rand((1, 512, self.n_embd))  # assuming input tensor of shape [batch, seq_length, n_embd]

        # Positive case: when attention_mask is None
        output = self.block.forward(x)
        self.assertEqual(output.shape, x.shape)

        # Negative case: when attention_mask is not None
        attention_mask = torch.ones((1, 512))  # attention mask of shape [batch, seq_length]
        output = self.block.forward(x, attention_mask)
        self.assertEqual(output.shape, x.shape)

        # Check attended tokens when masked (=0)
        attention_mask[0, 256:] = 0
        output_mask = self.block.forward(x, attention_mask)
        self.assertEqual(output_mask.shape, x.shape)

        self.assertTrue(torch.allclose(output[:, :256, :], output_mask[:, :256, :]))
        self.assertFalse(torch.allclose(output[:, 256:, :], output_mask[:, 256:, :]))


class TestBlock(TestCasualBlock):

    def setUp(self):
        # Setup process before each test
        self.n_embd = 128
        self.n_head = 2
        self.n_positions = 1024
        self.position_embedding = None
        self.is_causal = False
        self.block = gpt.Block(self.n_embd, self.n_head, self.n_positions, position_embedding=self.position_embedding,
                               is_causal=self.is_causal)


if __name__ == "__main__":
    unittest.main()
