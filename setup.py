from setuptools import setup, find_packages

setup(
    name='Face-Recognition',
    version='1.0',
    packages=find_packages(),
    description='Beschrijving van je project',
    author='Ralph Depondt',
    author_email='r.depondt@student.ou.nl',
    install_requires=[
        'numpy>=1.0',  # Vereist minimaal versie 1.0 van numpy
        'matplotlib>=3.0',  # Vereist minimaal versie 3.0 van matplotlib
        'pandas',
        'opencv-python',
        'face-recognition',
        'mediapipe'
    ],
)