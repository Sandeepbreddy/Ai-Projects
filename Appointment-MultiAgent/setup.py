from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    try:
        with open('req-packages.txt', 'r') as file:
            requirement_list = [
                line.strip() for line in file.readlines()
                if line.strip() and line.strip() != '-e .'
            ]
        return requirement_list
    except FileNotFoundError:
        print("requirements.txt file not found. Make sure it exists!")
        return []

setup(
    name="appointment-agentic",
    version="1.0.0",
    author="Sandeep Reddy",
    author_email="boda_sandeep01@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.10",  # Ensure compatible Python version
)