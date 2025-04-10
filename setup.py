from setuptools import find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()# Reads every line in the file
        # Removes the \n from the list of requirements as when new line is there \n will there so we need to remove it.
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='MLProject',
    version='0.0.1',
    author='vanya',
    author_email='xyz@abc.com',
    packages=find_packages(),
    install_requires=get_requirements('E:\udemy\Data_Science\requirements.txt')
)