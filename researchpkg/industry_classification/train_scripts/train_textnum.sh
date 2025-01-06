# 1.Textnum transformer using BGE base
python scripts/run_train_textnum_transformer.py -g count30_sic1agg_including_is_2023 -ptr bge_base -nt 0 -nh 8 -nl 4 -he 64 -ffw 256 -d 0 -e 60 -bs 128  -lr 0.0001 --max_tag_depth=5

# 2. Textnum transformer using Finlang
python scripts/run_train_textnum_transformer.py -g count30_sic1agg_including_is_2023 -ptr finlang -nt 0 -nh 8 -nl 4 -he 64 -ffw 256 -d 0 -e 60 -bs 128  -lr 0.0001 --max_tag_depth=5


# 3. Textnum transformer using Finlang with LRP modules (for explainability)
python scripts/run_train_textnum_transformer.py -g count30_sic1agg_including_is_2023 -ptr finlang -nt 0 -nh 8 -nl 4 -he 64 -ffw 256 -d 0 -e 60 -bs 128   -lr 0.0001 --max_tag_depth=5 --use_lrp_modules