from setuptools import setup

setup(
    name='corpus_downloader',
    version='',
    packages=['corpus_downloader'],
    url='',
    license='',
    author='Pierre PACI',
    author_email='',
    description='',
    entry_points={
        'console_scripts': ['earthlab-corpus=corpus_downloader.script:main']
    }, install_requires=['requests', 'Pillow', 'tqdm', 'h5py', 'numpy', 'scikit-image']
)
