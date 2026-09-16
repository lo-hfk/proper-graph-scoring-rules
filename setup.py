from setuptools import setup, find_packages

setup(
    name="propernetscore",
    version="0.1.0",
    description="Proper scoring rules for graph-learning tasks",
    author="Lorenz Hilfiker, Nathan Brack, Martin Sterchi",
    packages=find_packages(),  # Automatically finds the propernetscore/ package
    install_requires=[
        "numpy",
    ],
    extras_require={
        "networkx": ["networkx"],
        "igraph": ["python-igraph"],
        "dev": ["pytest", "jupyter"],
    },
    python_requires=">=3.8",
)