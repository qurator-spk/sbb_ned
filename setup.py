from io import open
from setuptools import find_packages, setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name="qurator-sbb-ned",
    version="0.0.1",
    author="The Qurator Team",
    author_email="qurator@sbb.spk-berlin.de",
    description="Qurator",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='qurator',
    license='Apache',
    url="https://qurator.ai",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=install_requires,
    entry_points={
      'console_scripts': [
        "per-sentence-ned-data=qurator.sbb_ned.cli:per_sentence_ned_data",
        "ned-pairing=qurator.sbb_ned.cli:ned_pairing",
        "ned-train-test-split=qurator.sbb_ned.cli:ned_train_test_split",
        "ned-features=qurator.sbb_ned.cli:ned_features",
        "ned-bert=qurator.sbb_ned.models.bert:main",
        "build-index=qurator.sbb_ned.cli:build",
        "build-context-matrix=qurator.sbb_ned.cli:build_context_matrix",
        "build-from-context-matrix=qurator.sbb_ned.cli:build_from_context_matrix",
        "evaluate-index=qurator.sbb_ned.cli:evaluate",
        "evaluate-with-context=qurator.sbb_ned.cli:evaluate_with_context",
        "evaluate-combined=qurator.sbb_ned.cli:evaluate_combined"
      ]
    },
    python_requires='>=3.6.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
