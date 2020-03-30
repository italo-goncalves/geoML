from setuptools import setup


# def readme():
#     with open('README.rst') as f:
#         return f.read()


setup(name='geoml',
      version='0.2.0',
      description='Machine learning for spatial data',
      #      long_description=readme(),
      keywords=['machine learning', 'spatial data', 'gaussian process'],
      url='http://github.com/italo-goncalves/geoML',
      author='Ítalo Gomes Gonçalves',
      author_email='italogoncalves.igg@gmail.com',
      license='GPL3',
      packages=['geoml'],
      package_dir={'geoml': 'src/geoml'},
      package_data={'geoml': ['sample_data/*.dat']},
      include_package_data=True,
      zip_safe=False,
      install_requires=['scikit-image', 'pandas', 'numpy',
                        'tensorflow', 'tensorflow-probability'])
