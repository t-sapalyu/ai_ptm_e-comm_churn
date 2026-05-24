"""Minimal setup.py so `pip install -e .` works with the src/ layout."""

from setuptools import find_packages, setup

setup(
    name="churn",
    version="0.1.0",
    description="RetailGenius customer churn prediction",
    author="Krishanth Dev, Thin Sapal Yu, Duy Thai, George Ebaidallah",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11,<3.12",
)
