from setuptools import setup, find_packages

setup(
    name="robustfsl",
    version="0.1",
    description="A robust tweak to FEw shot learning methods",
    author="Quentin Macé",
    author_email="quentinjg.mace@gmail.com",
    url="https://github.com/QuentinJGMace/RobustFSL",
    packages=find_packages(),
    install_requires=[
        "torch<2.0",
        "numpy",
        "loguru==0.5.3",
        "matplotlib",
        "PyYAML",
        "scikit-learn",
        "torchvision",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
