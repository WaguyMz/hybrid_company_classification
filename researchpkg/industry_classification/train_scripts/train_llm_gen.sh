
#1. LLM-gen with llama31_8b_instruct_gptq(Relative values template)
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_train_texttransformer_instruct_sft.py -ptr llama31_8b_instruct_gptq -bs 2  -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4 -e 15 -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5

#2. LLM-gen with llama31_8b_instruct_gptq(Raw values template)
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_train_texttransformer_instruct_sft_raw.py -ptr llama31_8b_instruct_gptq -bs 1 -lr 0.0001 -dp 0.5 --template_type RAW -ga 8 -e 15 -g count30_sic1agg_including_is_2023 -sl 1400 --max_tag_depth=5

#3.LLM-gen with fin_llama3(Relative values template)
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_train_texttransformer_instruct_sft.py -ptr fin_llama3 -bs 2  -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4 -e 10 -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5


#4.LLM-gen with fin_llama3(Raw values template)
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_train_texttransformer_instruct_sft_raw.py -ptr fin_llama3 -bs 1 -lr 0.0001 -dp 0.5 --template_type RAW -ga 8 -e 10 -g count30_sic1agg_including_is_2023 -sl 1400 --max_tag_depth=5




