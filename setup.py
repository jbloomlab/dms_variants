"""Setup script for ``dms_variants``."""


import platform
import re
import sys

from setuptools import Extension, setup


if not (sys.version_info[0] == 3 and sys.version_info[1] >= 6):
    raise RuntimeError(
                'dms_variants requires Python >=3.6.\n'
                f"You are using {sys.version_info[0]}.{sys.version_info[1]}.")

# get metadata from package `__init__.py` file as here:
# https://packaging.python.org/guides/single-sourcing-package-version/
metadata = {}
init_file = 'dms_variants/__init__.py'
with open(init_file) as f:
    init_text = f.read()
for dataname in ['version', 'author', 'email', 'url']:
    matches = re.findall(
            '__' + dataname + r'__\s+=\s+[\'"]([^\'"]+)[\'"]',
            init_text)
    if len(matches) != 1:
        raise ValueError(f"found {len(matches)} matches for {dataname} "
                         f"in {init_file}")
    else:
        metadata[dataname] = matches[0]

with open('README.rst') as f:
    readme = f.read()

extra_compile_args = []
if platform.system() != 'Windows':
    extra_compile_args.append('-Wno-error=declaration-after-statement')

# main setup command
setup(
    name='dms_variants',
    version=metadata['version'],
    author=metadata['author'],
    author_email=metadata['email'],
    url=metadata['url'],
    download_url='https://github.com/jbloomlab/dms_variants/tarball/' +
                 metadata['version'],  # tagged version on GitHub
    description='Analyze deep mutational scanning of barcoded variants.',
    long_description=readme,
    license='GPLv3',
    install_requires=[
            'biopython>=1.73',
            'matplotlib>=3.1',
            'pandas>=0.25.1',
            'plotnine!=0.7.0',  # https://github.com/has2k1/plotnine/issues/403
            'regex>=2.4.153',
            'requests',
            'scipy>=1.1.0',
            ],
    platforms='Linux and Mac OS X.',
    packages=['dms_variants'],
    package_dir={'dms_variants': 'dms_variants'},
    ext_modules=[
        Extension(
            'dms_variants._cutils',
            ['dms_variants/_cutils.c'],
            extra_compile_args=extra_compile_args,
            ),
        ],
    )
