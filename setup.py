import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='consensus-clustering',
    version='1.0.0',
    author='Michal Kazmierski',
    author_email='',
    description='A simple implementation of the consensus clustering algorithm.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mkazmier/consensus-clustering',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
