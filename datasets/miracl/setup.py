from setuptools import find_packages, setup

setup(
    name="miracl",
    packages=find_packages(exclude=["miracl_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
