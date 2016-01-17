import os
import sys
import subprocess

from setuptools import setup
from setuptools.command.test import test as TestCommand

def build_extensions():

    cwd = os.path.join(os.path.dirname(__file__),
                       'pyrustlearn/rustlearn-bindings')

    # Compile the Rust library
    subprocess.check_call(['cargo', 'build', '--release'],
                          cwd=cwd)

build_extensions()


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests/']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='pyrustlearn',
    version='0.1.0',
    packages=['pyrustlearn'],
    cmdclass={'test': PyTest},
    install_requires=['cffi', 'numpy', 'scipy', 'scikit-learn'],
    tests_require=['pytest'],
    package_data={'pyrustlearn': ['rustlearn-bindings/target/release/librustlearn.so']},
    author='Maciej Kula',
    license='Apache 2.0',
)
