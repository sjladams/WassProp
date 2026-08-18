import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WassProp",
    version="1.0.0.dev",
    author="Steven Adams, Eduardo Figueiredo",
    author_email="stevenjladams@gmail.com",
    description="Formal Uncertainty Propagation in Wasserstein Distance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjladams/WassProp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        'torch>=2.5.1',
        'discretize_distributions>=2.1.0',
        'pointwise_lipschitz'
        'bound_propagation>=0.4.6', 
        'POT>=0.9.6',
    ]
)