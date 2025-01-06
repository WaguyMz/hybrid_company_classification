from setuptools import find_packages, setup

requirements = """
gensim==4.3.1
joblib==1.3.1
nltk==3.8.1
numpy==1.25.1
pandas==2.0.3
pandas_parallel_apply==2.2
py_xbrl==2.2.8
pytorch_lightning
scikit_learn==1.3.0
torchmetrics
tqdm==4.65.0
tensorboard
tensorboardX
langchain_openai
langchain_core
matplotlib
transformers>=4.41.2
peft
lightgbm==4.0.0
xgboost==2.0.0
pyaml
GitPython
hyperopt
ray[tune]
treelib
seaborn
bitsandbytes==0.43.1
optimum
graphviz
auto-gptq
 markupsafe==2.0.1
trl
torch==2.3
"""


setup(
    name="researchpkg",
    author="anonymous",
    author_email="",
    description="",
    packages=find_packages(),
    install_requires=requirements,
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    scripts=[],
    zip_safe=False,
)
