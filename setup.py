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
        "ned-sentence-data=qurator.sbb_ned.cli:ned_sentence_data",
        "ned-pairing=qurator.sbb_ned.cli:ned_pairing",
        "ned-train-test-split=qurator.sbb_ned.cli:ned_train_test_split",
        "ned-pairing-examples=qurator.sbb_ned.cli:ned_pairing_examples",
        "ned-bert=qurator.sbb_ned.models.bert:main",

        "build-index=qurator.sbb_ned.cli:build",
        "build-context-matrix=qurator.sbb_ned.cli:build_context_matrix",
        "build-from-context-matrix=qurator.sbb_ned.cli:build_from_context_matrix",

        "evaluate-index=qurator.sbb_ned.cli:evaluate",
        "evaluate-with-context=qurator.sbb_ned.cli:evaluate_with_context",
        "evaluate-combined=qurator.sbb_ned.cli:evaluate_combined",

        "clef2tsv-2020=qurator.sbb_ned.ground_truth.clef_hipe_2020:cli_clef2tsv",
        "tsv2clef-2020=qurator.sbb_ned.ground_truth.clef_hipe_2020:cli_tsv2clef",

        "clef2tsv-2022=qurator.sbb_ned.ground_truth.clef_hipe_2022:cli_clef2tsv",
        "tsv2clef-2022=qurator.sbb_ned.ground_truth.clef_hipe_2022:cli_tsv2clef",

        "compute-knb-coverage-2020=qurator.sbb_ned.ground_truth.clef_hipe_2020:compute_knb_coverage",
        "compute-knb-coverage-2022=qurator.sbb_ned.ground_truth.clef_hipe_2022:compute_knb_coverage",

        "sentence-stat-2020=qurator.sbb_ned.ground_truth.clef_hipe_2020:sentence_stat",
        "sentence-stat-2022=qurator.sbb_ned.ground_truth.clef_hipe_2022:sentence_stat",

        "train-decider=qurator.sbb_ned.models.decider:train",
        "test-decider=qurator.sbb_ned.models.decider:test",

        "extract-normalization-table=qurator.sbb_ned.encoding.normalization:extract_normalization_table",
        "adapt-normalization-table=qurator.sbb_ned.encoding.normalization:adapt_table_to_corpus"
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
