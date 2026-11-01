from setuptools import find_packages, setup

package_name = "agripioneer_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/agripioneer.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="omnicampus-142210",
    description="AgriPioneer ROS 2 nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "simulator_node = agripioneer_ros.simulator_node:main",
            "perception_node = agripioneer_ros.perception_node:main",
            "planner_node = agripioneer_ros.planner_node:main",
            "controller_node = agripioneer_ros.controller_node:main",
        ],
    },
)
