from pathlib import Path

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext
from setuptools import setup


__version__ = '0.3'

CURRENT_DIRECTORY = Path(__file__).absolute().parent
SOURCES_DIRECTORY = CURRENT_DIRECTORY.parent.parent

LIBRARY_DIRECTORY = SOURCES_DIRECTORY / 'hnsw'
THIRD_PARTY_DIRECTORY = SOURCES_DIRECTORY / 'third_party'

with open(CURRENT_DIRECTORY / 'requirements.txt') as file:
    REQUIREMENTS = file.readlines()


ext_modules = [
    Pybind11Extension(
        'hnsw_index',
        ['bindings.cpp'],
        include_dirs=[LIBRARY_DIRECTORY, THIRD_PARTY_DIRECTORY],
        libraries=[],
        language='c++',
        extra_objects=[],
        extra_compile_args=['-fopenmp']
    ),
]


ParallelCompile("NPY_NUM_BUILD_JOBS").install()


setup(
    name='hnsw_index',
    version=__version__,
    description='hnsw_index',
    ext_modules=ext_modules,
    install_requires=REQUIREMENTS,
    cmdclass={'build_ext': build_ext},
    test_suite="tests",
    zip_safe=False,
)
