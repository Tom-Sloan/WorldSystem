from setuptools import setup, find_packages

setup(
    name="worldsystem-common",
    version="1.0.0",
    description="Shared utilities and configurations for WorldSystem microservices",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],  # No dependencies for config module
    author="WorldSystem Team",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)