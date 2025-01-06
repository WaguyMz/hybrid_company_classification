
#1. Run  LLm-gen with llama31_8b_instruct_gptq for testing. The model must be trained prior to this. This script can not work without training the model first.
~/.local/bin/accelerate launch --main_process_port=29823 scripts/run_test_llm_gen.py -ptr llama31_8b_instruct_gptq -bs 16 -lr 0.0001 -dp 0.5 --template_type DESCRIPTIVE -ga 4  -g count30_sic1agg_including_is_2023 -sl 1300 --max_tag_depth=5  --add_explanation

