#1. LLM-Clf with llama31_8b_gptq (relative values template)
python scripts/run_train_llm_clf.py -ptr llama31_8b_gptq -bs 2  -lr 0.0001 -dp 0.1 --template_type DESCRIPTIVE -ga 4 -g count30_sic1agg_including_is_2023 -sl 1000 -e15  --max_tag_depth=5

#2. LLM-Clf with llama31_8b_gptq (raw values template)
python scripts/run_train_llm_clf.py -ptr llama31_8b_gptq -bs 2  -lr 0.0001 -dp 0.1 --template_type RAW -ga 4 -g count30_sic1agg_including_is_2023 -sl 1200 -e15  --max_tag_depth=5

#3. LLM-Clf with fin_llama3 (relative values template)
python scripts/run_train_llm_clf.py -ptr fin_llama3 -bs 2  -lr 0.0001 -dp 0.1 --template_type DESCRIPTIVE -ga 4 -g count30_sic1agg_including_is_2023 -sl 1000 -e15  --max_tag_depth=5

#4. LLM-Clf with fin_llama3 (raw values template)
python scripts/run_train_llm_clf.py -ptr fin_llama3 -bs 2  -lr 0.0001 -dp 0.1 --template_type RAW -ga 4 -g count30_sic1agg_including_is_2023 -sl 1200 -e15  --max_tag_depth=5