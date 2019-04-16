from setuptools import setup
from os import path

# If Python3: Add "README.md" to setup. 
# Useful for PyPI (pip install wotan). Irrelevant for users using Python2
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

    
setup(name='wotan',
    version=1,
    description='Wotan is a free and open source algorithm to automagically remove stellar trends from light curves for exoplanet transit detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hippke/wotan',
    author='Michael Hippke',
    author_email='michael@hippke.org',
    license='MIT',
    packages=['wotan'],
    install_requires=[
        'numpy',
        'numba',
        ]
)
