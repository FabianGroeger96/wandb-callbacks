import os.path
import codecs

from setuptools import find_namespace_packages, setup
from os.path import abspath, dirname, join


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def parse_requirements(filename):
    """
    load requirements from a pip requirements file
    """
    line_iter = (line.strip() for line in open(filename))
    return [line for line in line_iter if line and not line.startswith("#")]


README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()

setup(name='wandb-callbacks',
      version=get_version('wandb_callbacks/__init__.py'),
      author='Fabian Groeger',
      author_email='fabian.groeger@bluewin.ch',
      description='Additional custom callbacks for Weights & Biases',
      long_description=README_MD,
      long_description_content_type="text/markdown",
      url='https://github.com/FabianGroeger96/wandb-callbacks',
      packages=find_namespace_packages(),
      keywords='wandb, keras, tensorflow, callbacks',
      install_reqs=parse_requirements('requirements.txt'),
      classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
          ])
