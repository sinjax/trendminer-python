#!/usr/bin/env python                                                                                                                                        
from setuptools import setup, find_packages

setup(
	name="Bivariate Model",
	version='0.1',
	description='An implementation of the user/word bivariate model',
	author='Sina Samangooei, Bill Lampos, Trevor',
	author_email='ss@ecs.soton.ac.uk',
	packages=find_packages(),
	install_requires=["numpy","scipy","ipython","h5py"],
	test_suite="bivariate.test",
)

