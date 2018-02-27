from setuptools import setup

setup(name='julia_env',
      version='0.1',
      description='Python Interface to Julia RL Environments',
      author='blake wulfe',
      author_email='blake.w.wulfe@gmail.com',
      license='MIT',
      packages=['julia_env'],
      zip_safe=False,
      install_requires=[
        'numpy',
      ])