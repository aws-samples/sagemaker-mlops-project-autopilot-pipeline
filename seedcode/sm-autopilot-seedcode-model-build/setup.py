import os
import setuptools

from pipelines.__version__ import (
    __title__,
    __description__,
    __version__,
    __author__,
    __author_email__,
    __license__,
    __url__,
)

here = os.path.abspath(os.path.dirname(__file__))


with open("README.md", "r") as f:
    readme = f.read()


required_packages = ["sagemaker==2.141.0", "scikit-learn==1.2.2"]
extras = {
    "test": [
        "black",
        "coverage",
        "flake8",
        "mock",
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "sagemaker",
        "tox",
        "scikit-learn",
    ]
}
setuptools.setup(
    name=__title__,
    description=__description__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=__url__,
    license=__license__,
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=required_packages,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "get-pipeline-definition=pipelines.get_pipeline_definition:main",
            "run-pipeline=pipelines.run_pipeline:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
