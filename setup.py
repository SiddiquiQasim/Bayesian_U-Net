from setuptools import find_packages, setup

setup(
    name='bayesian_unet',
    packages=find_packages(),
    version='0.1.0',
    description='visualize uncertainty in semantic segmentation problems',
    author='Qasim',
    license='MIT',
    install_requires=[
        'tqdm==4.64.1',
        'PyYAML==6.0',
        'numpy==1.23.3',
        'tensorflow==2.9.0',
        'tensorflow-probability==0.18.0',
        'matplotlib==3.5.3'
    ]
)