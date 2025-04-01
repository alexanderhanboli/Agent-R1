from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-r1",
    version="0.1.0",
    author="hanboli",
    description="A reinforcement learning agent framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hanboli/Agent-R1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "agent_r1": ["src/config/*.yaml"],
    },
) 