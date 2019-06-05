from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sort_attributes',
    ext_modules=[
        CUDAExtension('sort_attributes', [
            'sort_attributes.cpp',
            'sort_attributes_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
