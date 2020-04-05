from setuptools import setup, find_packages
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'car_racing_variants'))
from version import VERSION

# Don't import gym module here, since deps may not be installed
for package in find_packages():
    if '_gym_' in package:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), package))

setup(
    name='car_racing_variants',
    version=VERSION,
    description='CarRacing variants that were used in the paper Neuroevolution of Self-Interpretable Agents.',
    url='https://github.com/google/brain-tokyo-workshop/car-racing-variants',
    author='Yujin Tang',
    author_email='yujintang@google.com',
    license='MIT License',
    zip_safe=False,
    install_requires=['gym[box2d]>=0.15.3',
                      'opencv-python==4.1.2.30'],
)
