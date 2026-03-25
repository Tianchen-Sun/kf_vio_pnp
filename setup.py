from setuptools import find_packages, setup

package_name = 'kf_vio_pnp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/kf_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/kf_vio_gate.python.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tlab-uav',
    maintainer_email='tianchen.sun787@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'kf_node=kf_vio_pnp.kf_node:main',
    ],
},
)
