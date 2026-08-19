import unittest

from app import (
    AMD_GPU_OPTION,
    NVIDIA_GPU_OPTION,
    PROCESSING_OPTIONS,
    normalize_gpu_backend,
)


class GpuBackendTests(unittest.TestCase):
    def test_gpu_options_include_amd_and_nvidia(self) -> None:
        self.assertIn(AMD_GPU_OPTION, PROCESSING_OPTIONS)
        self.assertIn(NVIDIA_GPU_OPTION, PROCESSING_OPTIONS)

    def test_normalize_gpu_backend_accepts_common_names(self) -> None:
        self.assertEqual(normalize_gpu_backend("amd"), "amd")
        self.assertEqual(normalize_gpu_backend("AMD"), "amd")
        self.assertEqual(normalize_gpu_backend("directml"), "amd")
        self.assertEqual(normalize_gpu_backend("nvidia"), "nvidia")
        self.assertEqual(normalize_gpu_backend("cuda"), "nvidia")
        self.assertEqual(normalize_gpu_backend("GPU NVIDIA (CUDA)"), "nvidia")


if __name__ == "__main__":
    unittest.main()
