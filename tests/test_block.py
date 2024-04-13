import unittest
from models import gpt
import torch
from torch import nn


class TestBlock(unittest.TestCase):

    def setUp(self):
        # Setup process before each test
        self.n_embd = 2
        self.n_head = 1
        self.n_positions = 256
        self.sequence_length = 4
        self.position_embedding = None
        self.is_causal = False
        self.block = gpt.Block(self.n_embd, self.n_head, self.n_positions, position_embedding=self.position_embedding,
                               is_causal=self.is_causal)
        self.block.eval()

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
        x = torch.rand((1, self.sequence_length, self.n_embd))  # input tensor of shape [batch, seq_length, n_embd]

        # Positive case: when attention_mask is None
        output = self.block.forward(x)
        self.assertEqual(output.shape, x.shape)

        # Negative case: when attention_mask is not None (all attended so outputs should be the same)
        attention_mask = torch.ones((1, self.sequence_length))  # attention mask of shape [batch, seq_length]
        output_mask = self.block.forward(x, attention_mask)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.allclose(output, output_mask))

        # Check attended output is same when non-attended tokens change (=0)
        attention_mask = torch.ones((1, self.sequence_length))  # attention mask of shape [batch, seq_length]
        attention_mask[0, int(self.sequence_length / 4):int(self.sequence_length / 2)] = 0
        output = self.block.forward(x, attention_mask)
        x[0, int(self.sequence_length / 4):int(self.sequence_length / 2), :] = 0
        output_x0 = self.block.forward(x, attention_mask)
        self.assertTrue(torch.allclose(output[0, :int(self.sequence_length / 4):, :],
                                       output_x0[0, : int(self.sequence_length / 4):, :]))
        self.assertTrue(torch.allclose(output[0, int(self.sequence_length / 2):, :],
                                       output_x0[0, int(self.sequence_length / 2):, :]))


class TestCasualBlock(TestBlock):

    def setUp(self):
        # Setup process before each test
        self.n_embd = 2
        self.n_head = 1
        self.n_positions = 256
        self.sequence_length = 4
        self.position_embedding = None
        self.is_causal = True
        self.block = gpt.Block(self.n_embd, self.n_head, self.n_positions, position_embedding=self.position_embedding,
                               is_causal=self.is_causal)
        self.block.eval()



if __name__ == "__main__":
    unittest.main()