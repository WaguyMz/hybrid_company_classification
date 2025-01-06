#1. Run LLM-gen with llama31_8b_instruct_gptq (Relative values template) in zero-shot setting. Also generate explanations.
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_test_llm_gen.py -ptr llama31_8b_instruct_gptq -bs 16 -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4  -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5  --add_explanation --zero_shot

#2. Zero shot eval with fin_llama3
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_test_llm_gen.py -ptr fin_llama3 -bs 16 -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4   -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5  --add_explanation --zero_shot