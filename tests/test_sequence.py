import torch
import unittest
from CATS.layers.sequence import SequencePoolingLayer
from torch.autograd import Variable

class TestSequencePoolingLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.seq_pooling_layer = SequencePoolingLayer('mean', True, 'cpu')

    def test_init(self):
        self.assertEqual('mean', self.seq_pooling_layer.mode)
        self.assertEqual(True, self.seq_pooling_layer.supports_masking)
        self.assertEqual(torch.FloatTensor([1e-8]).to('cpu'), self.seq_pooling_layer.eps)

    def test_forward(self):
        seq_embed_list = Variable(torch.randn(5, 1, 3))
        user_behavior_length = Variable(torch.randn(5, 1))
        input_data = [seq_embed_list, user_behavior_length]
        result = self.seq_pooling_layer.forward(input_data)
        
        self.assertEqual(result.shape, (5, 1, 3))
    
    def test_invalid_forward(self):
        with self.assertRaises(ValueError):
            tensor2d = torch.randn(2,2)
            tensor1d = torch.randn(2)
            input_data = [tensor2d, tensor1d]
            result = self.seq_pooling_layer.forward(input_data)

if __name__ == "__main__":
    unittest.main()