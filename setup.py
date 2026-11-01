from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'muzammil_controller'

setup(
    name=package_name,
    version='1.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Muzammil Ali',
    maintainer_email='muzammilalimkd@gmail.com',
    description='Horizon-weighted disparity gap autonomous racing controller for the CIE-KFUPM F1TENTH competition.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = muzammil_controller.controller_node:main',
        ],
    },
)
