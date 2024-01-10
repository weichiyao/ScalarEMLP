from setuptools import setup,find_packages
import sys, os, re

README_FILE = 'README.md'

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

project_name = "emlp"
setup(name=project_name,
      description="Scalar-Based Equivariant Machine Learning Methods",
      version= get_property('__version__',project_name),
      author='Weichi Yao, Marc Finzi',
      author_email='weichiy@umich.edu, maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.8',
      install_requires=['h5py','objax','pytest','plum-dispatch',
            'optax','tqdm>=4.38','pytorch-lightning','lightning'],
      extras_require = {
          'EXPTS':['olive-oil-ml']
      },
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/weichiyao/ScalarEMLP',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords=[
            'equivariance','MLP','symmetry','group','AI','neural network',
            'representation','group theory','deep learning','machine learning',
            'rotation','Lorentz invariance','scalar-based'
      ],

)
