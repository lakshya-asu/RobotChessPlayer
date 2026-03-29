from setuptools import find_packages, setup
from pathlib import Path
import os
from glob import glob

package_name = 'chess_manipulator'
package_root = Path(__file__).resolve().parent

models_paths = []
models_root = package_root / 'models'
if models_root.exists():
    for root, _, files in os.walk(models_root):
        if not files:
            continue
        relative_root = Path(root).relative_to(package_root)
        install_root = Path('share') / package_name / relative_root
        source_files = [str(relative_root / filename) for filename in files]
        models_paths.append((str(install_root), source_files))
    
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        #Add all xacro and URDF files that descripe the robot to the share directory
        (os.path.join('share',package_name,'description'),glob('description/*.xacro')),
        #Adding visual and collision mehses to the share directory
        (os.path.join('share',package_name,'meshes','visual'),glob('meshes/visual/*.dae')),
        (os.path.join('share',package_name,'meshes','collision'),glob('meshes/collision/*.stl')),
        #Adding Rviz2 configuration files
        (os.path.join('share',package_name,'rviz'),glob('rviz/*.rviz')),
        #Adding launch files
        (os.path.join('share',package_name,'launch'),glob('launch/*.launch.*')),
        (os.path.join('share',package_name,'launch'),glob('launch/*.py')),
        (os.path.join('share',package_name,'config'),glob('config/*.yaml')),
        (os.path.join('share',package_name,'config'),glob('config/*.config')),
        (os.path.join('share',package_name,'worlds'),glob('worlds/*')),
        (os.path.join('lib',package_name), glob('chess_manipulator/RobotClass.py'))
        ]+models_paths,
    
    install_requires=['setuptools', 'numpy', 'PyYAML', 'chess'],
    zip_safe=True,
    maintainer='Zein Alabedeen Barhoum and Rahaf Alshaowa',
    maintainer_email='zein.barhoum799@gmail.com',
    description='Franka Emika Panda manipulator playing chess',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = chess_manipulator.controller:main',
            'multi_point_controller = chess_manipulator.multi_point_controller:main',
            'example_game = chess_manipulator.example_game:main',
            'board_perception = chess_manipulator.nodes.board_perception:main',
            'chess_manager = chess_manipulator.nodes.chess_manager:main',
            'game_coordinator = chess_manipulator.coordinator.game_coordinator:main',
            'isaac_bridge = chess_manipulator.nodes.isaac_bridge:main',
            'joint_state_filter = chess_manipulator.nodes.joint_state_filter:main',
            'rl_infer = chess_manipulator.rl.infer:main',
            'rl_train = chess_manipulator.rl.train:main',
            'robot_executor = chess_manipulator.nodes.robot_executor:main',
            'ros_gz_trajectory_controller = chess_manipulator.nodes.ros_gz_trajectory_controller:main',
            'trajectory_relay = chess_manipulator.nodes.trajectory_relay:main',
            'demo_turn = chess_manipulator.tools.demo_turn:main',
        ],
    },
)
