from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torch_persistent_homology',
      ext_modules=[cpp_extension.CppExtension(
        'torch_persistent_homology',
        ['perisistent_homology_cpu.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})