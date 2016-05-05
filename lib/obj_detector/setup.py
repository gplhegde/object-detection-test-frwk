#import numpy as np
#from distutils.core import setup
#from Cython.Build import cythonize

#setup(
#  name = 'lbp calculator',
#  ext_modules = cythonize("lbp_feat.pyx"),
#  include_dirs=[np.get_include()]
#)

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = [
    Extension(
        "lbp_feat",
        ["lbp_feat.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O3", "-Ofast"],
    ),
]
cmdclass.update({'build_ext': build_ext})

setup(
    name='lbp_feat_gen',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
