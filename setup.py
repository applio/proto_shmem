#!/usr/bin/env python3

import setuptools

import sys
if sys.version_info < (3, 6):
    raise NotImplementedError("Sorry, you need at least Python 3.6+ to use this.")

from mmap import PAGESIZE  # TODO: Can pull this straight from Python.h and thus eliminate probe_results.h?

VERSION = "1.0.0"

name = "posixshmem"
description = "POSIX IPC primitives (shared memory) for Python"
#with open("README") as f:
#    long_description = f.read().strip()
author = "Davin Potts / Philip Semanchuk"
author_email = "python@discontinuity.net"
maintainer = "Davin Potts"
url = "https://github.com/applio/posixshmem/"
source_files = ["posixshmem.c"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: BSD :: FreeBSD",
    "Operating System :: POSIX :: Linux",
    "Operating System :: POSIX :: SunOS/Solaris",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Utilities"
]
license = "http://creativecommons.org/licenses/BSD/"
keywords = "ipc inter-process communication shared memory shm"

libraries = []

# TODO: Linux & FreeBSD require linking against the realtime libs
# This causes an error on other platforms
if sys.platform.startswith("bsd"):
    libraries.append("rt")

ext_modules = [
    setuptools.Extension(
        "posixshmem",
        source_files,
        libraries=libraries,
        depends=[
            "posixshmem.c",
            "probe_results.h",
        ],
        # extra_compile_args=['-E']
    )
]

include_file_contents = f"""
/*
This header file was generated when you ran setup. Once created, the setup
process won't overwrite it, so you can adjust the values by hand and
recompile if you need to.

On your platform, this file may contain only this comment -- that's OK!

To recreate this file, just delete it and re-run setup.py.
*/

#define POSIXSHMEM_VERSION      "{VERSION}"
#ifndef PAGE_SIZE
#define PAGE_SIZE               {PAGESIZE}
#endif
"""

py_modules = [
    "shared_memory",
    "proto_shmem"
]


if __name__ == '__main__':
    with open("probe_results.h", "wt") as pr:
        pr.write(include_file_contents)

    setuptools.setup(
        name=name,
        version=VERSION,
        description=description,
        #long_description=long_description,
        author=author,
        author_email=author_email,
        maintainer=maintainer,
        url=url,
        classifiers=classifiers,
        license=license,
        keywords=keywords,
        py_modules=py_modules,
        ext_modules=ext_modules
    )
