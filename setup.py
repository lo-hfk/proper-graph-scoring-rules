from setuptools import setup, find_packages

setup(
    name="propnetscore",
    version="0.1.0",
    description="Proper scoring rules for graph-learning tasks",
    author="Lorenz Hilfiker, Nathan Brack, Martin Sterchi",
    packages=find_packages(),  # automatically finds the propernetscore/ package
    install_requires=[
        "numpy", "scipy"
    ],
    extras_require={
        "julia": ["juliacall"],
        "dev": ["pytest", "jupyter", "networkx"],
    },
    python_requires=">=3.8",
)