from setuptools import find_packages, setup

package_name = "perception_tower_sensor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/turntable_params.yaml"]),
        ("share/" + package_name + "/launch", ["launch/sensor_env.launch.py"]),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="perception_tower",
    maintainer_email="dev@example.com",
    description="Turntable control node",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "turntable_node = perception_tower_sensor.turntable_node:main",
        ],
    },
)
