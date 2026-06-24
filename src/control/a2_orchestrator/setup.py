import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'a2_orchestrator'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

for path in glob('launch/*.launch.py'):
    data_files.append(
        (os.path.join('share', package_name, 'launch'), [path])
    )

for path in glob('config/*.yaml'):
    data_files.append(
        (os.path.join('share', package_name, 'config'), [path])
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dino',
    maintainer_email='dino@todo.todo',
    description='Orchestration Behavior Tree package for A2 robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'velocity_publisher = a2_orchestrator.velocity_publisher:main',
            'waypoint_mux = a2_orchestrator.waypoint_mux:main',
            'mission_orchestrator = a2_orchestrator.mission_orchestrator:main',
            'detection_processor = a2_orchestrator.detection_processor:main',
        ],
    },
)
