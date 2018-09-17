# Python-ish modules
# setuptools is apparently distributed with python.org Python now. Does that mean it's
# standard? Who knows. I need it to build wheels on my machine, otherwise I can get by just
# fine with distutils.
try:
    import setuptools as distutools
except ImportError:
    import distutils.core as distutools

import sys
from mmap import PAGESIZE  # TODO: Can pull this straight from Python.h and thus eliminate probe_results.h?

VERSION = "1.0.0"

name = "posixshmem"
description = "POSIX IPC primitives (semaphores, shared memory and message queues) for Python"
#with open("README") as f:
#    long_description = f.read().strip()
author = "Philip Semanchuk"
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
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Utilities"
]
license = "http://creativecommons.org/licenses/BSD/"
keywords = "ipc inter-process communication semaphore shared memory shm message queue"

libraries = []

# TODO: Linux & FreeBSD require linking against the realtime libs
# This causes an error on other platforms
if sys.platform.startswith("bsd"):
    libraries.append("rt")

ext_modules = [
    distutools.Extension(
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


if __name__ == '__main__':
    with open("probe_results.h", "wt") as pr:
        pr.write(include_file_contents)

    distutools.setup(
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
        ext_modules=ext_modules
    )
