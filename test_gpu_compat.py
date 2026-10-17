import unittest
from unittest.mock import patch
from types import SimpleNamespace
import gpu_compat
from backend import patched_source

class GPUChecks(unittest.TestCase):
    def test_profiles_cover_new_and_old_devices(self):
        for cap,expected in [([12,0],'cu128'),([10,0],'cu128'),([8,9],'cu121'),([6,1],'cu121'),([5,0],'cu121')]:
            self.assertEqual(gpu_compat.choose_profile({'devices':[{'capability':cap}]}),expected)
    def test_detection_falls_back_to_driver_inventory(self):
        self.assertEqual(gpu_compat.choose_profile({'nvidia_smi':{'devices':[{'capability':[12,0]}]}}),'cu128')
    def test_unsupported_or_unknown_stops(self):
        for report in [{},{'devices':[{'capability':[3,5]}]}]:
            with self.assertRaises(ValueError):gpu_compat.choose_profile(report)
    @patch('gpu_compat.subprocess.run')
    def test_nvidia_driver_csv(self,run):
        run.return_value=SimpleNamespace(returncode=0,stdout='NVIDIA GeForce RTX 5090, 12.0, 576.00\n',stderr='')
        r=gpu_compat.nvidia_inventory()
        self.assertEqual(r['devices'][0]['capability'],[12,0])
    @patch('gpu_compat.subprocess.run')
    def test_child_probe_protocol(self,run):
        run.return_value=SimpleNamespace(returncode=0,stdout='extra library output\n{"ok":true,"devices":[]}\n',stderr='warning')
        r=gpu_compat.isolated_probe()
        self.assertTrue(r['ok']);self.assertEqual(r['warnings'],'warning')
    @patch('gpu_compat.subprocess.run')
    @patch('gpu_compat.nvidia_inventory',return_value={})
    def test_crashed_child_is_failure(self,inventory,run):
        run.return_value=SimpleNamespace(returncode=1,stdout='',stderr='CUDA failure')
        self.assertFalse(gpu_compat.isolated_probe()['ok'])
    def test_upstream_checkpoint_compatibility(self):
        for device in ['cpu','cuda']:
            source=patched_source(device)
            compile(source,'<patched>','exec')
            self.assertIn('torch.load(model_file, map_location=self.device, weights_only=False)',source)
    @patch('repair_cuda.subprocess.run')
    def test_repair_installs_matching_cuda_wheels(self,run):
        from repair_cuda import install_profile
        install_profile('cu128')
        command=run.call_args.args[0]
        self.assertIn('torch==2.7.1',command)
        self.assertIn('torchaudio==2.7.1',command)
        self.assertEqual(command[-1],'https://download.pytorch.org/whl/cu128')
        self.assertIn('--force-reinstall',command)
if __name__=='__main__':unittest.main()
