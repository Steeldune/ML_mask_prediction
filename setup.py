from setuptools import setup

DISTNAME = 'mask-prediction'
DESCRIPTION = 'mask-prediction: ML-based segmentation of biomedical images'
MAINTAINER = 'Simon van Staalduine'
MAINTAINER_EMAIL = 'S.E.vanStaalduine@student.tudelft.nl'
LICENSE = 'LICENSE'
README = 'README.md'
URL = 'https://github.com/Steeldune/ML_mask_prediction'
VERSION = '0.1.dev'
PACKAGES = ['mask-prediction']
INSTALL_REQUIRES = [
    # 'tensorflow',
]

if __name__ == '__main__':

    setup(name=DISTNAME,
          version=VERSION,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          packages=PACKAGES,
          include_package_data=True,
          url=URL,
          license=LICENSE,
          description=DESCRIPTION,
          long_description=open(README).read(),
          install_requires=INSTALL_REQUIRES)
