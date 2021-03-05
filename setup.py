# 1. Upload to PyPI:
# (Note: Building 'bdist_wheel' requires package 'wheel')
# python3 setup.py sdist bdist_wheel
# twine upload dist/*
#
# 2. Check if everything looks all right: https://pypi.python.org/pypi/SoMeWeTa
#
# 3. Go to https://github.com/tsproisl/SoMeWeTa/releases/new and
# create a new release

from os import path
from setuptools import setup

version = {}
with open("someweta/version.py") as fh:
    exec(fh.read(), version)

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as fh:
    long_description = fh.read()

setup(
    name='SoMeWeTa',
    version=version["__version__"],
    author='Thomas Proisl',
    author_email='thomas.proisl@fau.de',
    packages=[
        'someweta',
    ],
    scripts=[
        'bin/somewe-tagger',
    ],
    url="https://github.com/tsproisl/SoMeWeTa",
    download_url='https://github.com/tsproisl/SoMeWeTa/archive/v%s.tar.gz' % version["__version__"],
    license='GNU General Public License v3 or later (GPLv3+)',
    description='A part-of-speech tagger with support for domain adaptation and external resources.',
    long_description=long_description,
    install_requires=[
        "numpy",
        "regex>=2019.02.18",
    ],
    python_requires='>=3.4',
    classifiers=[
        # 'Development Status :: 3 - Alpha',
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Natural Language :: French',
        'Natural Language :: German',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
    ],
)
