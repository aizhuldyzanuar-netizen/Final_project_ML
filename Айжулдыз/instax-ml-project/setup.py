from setuptools import setup, find_packages

setup(
    name='instax-ml-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A machine learning project for analyzing sales transaction data.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'ipywidgets',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)