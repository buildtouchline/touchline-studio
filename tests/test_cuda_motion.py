"""CUDA admission guards; real CUDA parity is an explicit operator-only tool."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from touchline.cuda_motion import CudaMotionPredictor


class CudaMotionGuards(unittest.TestCase):
    def test_cuda_request_fails_closed_without_device(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with patch.dict(sys.modules,{'torch':fake}):
            with self.assertRaisesRegex(RuntimeError,'CUDA is unavailable'):
                CudaMotionPredictor(object())

    def test_cuda_request_does_not_accept_cpu_motion_tensor(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),float32='fixture',
            as_tensor=lambda *args,**kwargs:SimpleNamespace(device=SimpleNamespace(type='cpu')))
        with patch.dict(sys.modules,{'torch':fake}):
            with self.assertRaisesRegex(RuntimeError,'not on CUDA'):
                CudaMotionPredictor(SimpleNamespace(_motion_mat=[]))
