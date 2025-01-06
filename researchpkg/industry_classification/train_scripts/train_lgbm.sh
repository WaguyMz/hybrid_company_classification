
# Lgbm with relative values representation
python scripts/run_train_lgbm_classifier.py -g  count30_sic1agg_including_is_2023 --normalization local --max_depth 5

# Lgbm with raw values representation
python scripts/run_train_lgbm_classifier.py -g  count30_sic1agg_including_is_2023 --normalization none --max_depth 5