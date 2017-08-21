import io
import os
import sys
from setuptools import setup, find_packages


if sys.version_info[:2] < (3, 6):
    raise Exception('This version of tensorplay needs Python 3.6 or later.')


def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()


REQUIRED_PACKAGES = [
    'numpy >= 1.13.1'
]

EXTRA_PACKAGES = {
    'cpu': ['tensorflow>=1.3.0'],
    'gpu': ['tensorflow-gpu>=1.3.0']
}

TEST_PACKAGES = [
    'pytest',
    'pylint'
]

setup(
    name='tensorplay',
    version='0.1',
    description='Tensorflow playground',
    long_description=readfile('README.md'),
    packages=find_packages(),
    author='Simon Ho',
    author_email='bawongfai@gmail.com',
    keywords='tensorflow',
    platforms='any',
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    setup_requires=['pytest-runner', 'pytest-pylint'],
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    tests_require=TEST_PACKAGES,
    test_suite="tests",
    include_package_data=True
)
