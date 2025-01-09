"""Setup configuration for MrBoB Agent Tools."""

from setuptools import setup, find_packages

setup(
    name="mrbob-agent-tools",
    version="0.1.0",
    description="Insurance agent tools for policy management and reporting",
    author="Laith Abood",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "cachetools>=5.0.0",
        "tenacity>=8.0.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
        "openpyxl>=3.0.0",
        "reportlab>=3.6.0",
        "xlsxwriter>=3.0.0"
    ],
    entry_points={
        "console_scripts": [
            "mrbob=mrbob.cli:main",
        ],
    },
)
