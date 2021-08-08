from setuptools import setup

setup(
    name='gsdmm',
    packages=['gsdmm'],
    version=0.2,
    description='GSDMM: Short text clustering',
    license='MIT',
    install_requires= ['numpy','sklearn'],
	extras_require= {
        'plot': ['matplotlib','wordcloud']
    }
)
