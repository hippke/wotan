from setuptools import setup
from os import path

# Pull wotan version from single source of truth file
try:  # Python 2
    execfile(path.join("wotan", 'version.py'))
except:  # Python 3
    exec(open(path.join("wotan", 'version.py')).read())

# If Python3: Add "README.md" to setup. 
# Useful for PyPI (pip install wotan). Irrelevant for users using Python2
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

setup(
    name='wotan',
    version=WOTAN_VERSIONING,
    description='Detrending algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hippke/wotan',
    author='Michael Hippke',
    author_email='michael@hippke.org',
    license='MIT',
    zip_safe=False,
    packages=['wotan'],
    install_requires=[
        'numpy',
        'numba',
        'scipy'
        ]
)