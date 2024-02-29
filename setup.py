from pathlib import Path

from setuptools import setup, find_packages


def parse_requirements(file_path):
    requirements = []
    for x in Path(file_path).read_text().split("\n"):
        x = x.strip()
        if x and not x.startswith("#"):
            requirements.append(x.split("#")[0].strip())  # ignore inline comments
    return requirements


setup(
    name="zjmod",
    version="1.0",
    author="TongZJ",
    author_email="1400721986@qq.com",

    description="nothing...",
    url="https://github.com/Instinct323/mod",
    packages=find_packages(),

    setup_requires=[],
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.8"
)
