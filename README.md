# README


This repository contains the code related to the paper : *Linking Industry Sectors and Financial Statements: A Hybrid Approach for
Company Classification*


## 1. Setup 


- We highly recommend using a virtual environment to run the code.
- To install the required packages, run the following command from the root directory
```bash
pip install . -e
```
- To run LLM-gen and LLM-clf scripts, ensure you have a cuda compatible GPU with at least 40gb of memory. Nvidia A100 is recommended.
That will install all required packages to run the code. the flag -e install the package in editable mode, so you can modify the code and the changes will be reflected in the package.


## 2. Data


- The preprocesed data is available in the `data/sec_data_v2` directory.
- No need to relaunch any preprocessing script, the data is ready to be used.


## 3. Training

- Python training scripts are available in the `researchpkg/industry_classification/scripts` directory.
- We also prepare scrpits to rerun the experiments presented in the paper. They are available in the `researchpkg/industry_classification/train_scripts` and
`researchpkg/industry_classification/test_scripts` directories.
- To train the model, open of the training scripts and copy&paset the scripts line by line in the terminal, each line corresponds to a different experiment.

## Example of Training commands

### Training LightGBM with relatives values:

```bash
python scripts/run_train_lgbm_classifier.py -g  count30_sic1agg_including_is_2023 --normalization local --max_depth 5
```

### Training LLM-clf with relatives representation.

```bash
python scripts/run_train_llm_clf.py -ptr llama31_8b_gptq -bs 2  -lr 0.0001 -dp 0.1 --template_type DESCRIPTIVE -ga 4 -g count30_sic1agg_including_is_2023 -sl 1000 -e15  --max_tag_depth=5
```

### Training LLM-gen with relatives representation

```bash
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_train_texttransformer_instruct_sft.py -ptr llama31_8b_instruct_gptq -bs 2  -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4 -e 15 -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5
```

### Training Text-Numeric Network using FinLang embedding

```bash
python scripts/run_train_textnum_transformer.py -g count30_sic1agg_including_is_2023 -ptr finlang -nt 0 -nh 8 -nl 4 -he 64 -ffw 256 -d 0 -e 60 -bs 128  -lr 0.0001 --max_tag_depth=5
```


## 4. Testing


Except for LLM-gen, all models test evaluation is done in the training script.
Regarding LLM-gen, the testing script is available in the `researchpkg/industry_classification/test_scripts` directory.

### Testing LLM-gen

```bash
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_test_llm_gen.py -ptr llama31_8b_instruct_gptq -bs 16 -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4  -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5  --add_explanation
```
The argument `--add_explanation` is optional, it will generate the explanation for the predictions. The output are saved in experiments/XXX/df_results_tests.csv and experiments/XXX/df_results_tests_with_explanations.csv


### LLM-gen zero-shot evaluation

The zero evaluation script is available in the `researchpkg/industry_classification/test_scripts/test_llm_gen_zero_shot.sh` file.

```bash
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_test_llm_gen.py -ptr llama31_8b_instruct_gptq -bs 16 -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4  -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5  --add_explanation --zero_shot
```


### 5. Text-Numeric Explainability
A notebook is provided for compute the relevance scores of the text-numeric model. The notebook is available in the `researchpkg/industry_classification/notebooks/textnum_transformer_explainability_lrp.ipynb
` directory. Prior to running, ensure you train the model with --lrp_modules flag as provided in the file `scripts/train_textnum.sh`
