import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


version = "0.0.0"

setup(
    name="Fanan 🎨 💗",
    version=version,
    description="Art & Creativity in JAX 🎨 💗",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmrMKayid/fanan",
    author="Amr Kayid",
    author_email="amrmkayid@gmail.com",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
