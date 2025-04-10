from setuptools import setup, find_packages

setup(
    name="neuralm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "pyyaml>=5.1",
    ],
    author="Igor Sadoune",
    author_email="igor.sadoune@pm.me",
    description="A PyTorch model generator from YAML configurations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="pytorch, deep learning, neural networks, yaml",
    url="https://github.com/IgorSadoune/neuralm",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)