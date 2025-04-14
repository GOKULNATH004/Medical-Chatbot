from setuptools import find_packages, setup

setup(
    name='Generative AI Project',
    version='0.0.1',
    author='Gokulnath',
    author_email='gokulnath0804@gmail.com',
    packages=find_packages(),
    install_requires=[
        'flask',
        'python-dotenv',
        'langchain',
        'langchain-community',
        'langchain-huggingface',
        'langchain-ollama',
        'PyPDF2'
    ],
)