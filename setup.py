from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name="music_generation_rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-agent system for symbolic music generation using reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/music-generation-rl",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "mido>=1.2.10,<2.0.0",
        "pretty_midi>=0.2.10,<0.3.0",
        "muspy>=0.5.0,<0.6.0",
        "torch>=2.0.0,<3.0.0",
        "tianshou>=2.0.0,<3.0.0",
        "gymnasium>=1.0.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "seaborn>=0.12.0,<1.0.0",
        "plotly>=5.0.0,<6.0.0",
        "networkx>=3.0.0,<4.0.0",
        "jsonschema>=4.0.0,<5.0.0",
        "wandb>=0.15.0,<1.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train-music=scripts.train:main",
            "evaluate-music=scripts.evaluate:main",
        ],
    },
)
