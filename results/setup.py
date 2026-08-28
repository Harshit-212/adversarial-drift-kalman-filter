from setuptools import find_packages, setup

setup(
    name="adrift",
    version="1.0.0",
    description="Adversarial Drift in Sequential Inference Systems: A Hidden Failure Mode in Bayesian Estimators",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "jupyter>=1.0", "notebook>=7.0"],
    },
)
