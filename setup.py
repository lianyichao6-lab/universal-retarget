from setuptools import setup, find_packages

setup(
    name="anydexretarget",
    version="0.2.0",
    description="High-precision hand pose retargeting system for Shadow Hand using adaptive analytical optimization",
    python_requires=">=3.10",
    license="MIT",
    packages=find_packages(),
    package_data={
        "anydexretarget": [],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "nlopt",
        "pin>=3.8.0",
        "pyyaml",
    ],
)
