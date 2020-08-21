from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dnn_from_scratch",
    version="0.1.dev1",
    author="Shivam Shrirao",
    author_email="shivamshrirao@gmail.com",
    description="A high level deep learning library for Convolutional Neural Networks,GANs and more, made from scratch(numpy/cupy implementation).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShivamShrirao/dnn_from_scratch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 1 - Planning",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires='>=3.6',
)