import os

import torch

from researchpkg.industry_classification.config import (
    MAX_CORE_USAGE,
    SEC_ROOT_DATA_DIR,
)
from researchpkg.industry_classification.dataset.sec_transformer_datamodule import (
    SecTextTransformerDataset,
)
from researchpkg.industry_classification.dataset.utils import DatasetType


class SftDatasetProcessor:
    def __init__(
        self,
        source_dataset: SecTextTransformerDataset,
        output_file: str,
        nb_workers=MAX_CORE_USAGE,
        batch_size=32,
    ):
        self.source_dataset = source_dataset
        self.output_file = output_file
        self.nb_workers = nb_workers
        self.batch_size = batch_size

    def extract_all(self):
        """
        Extract all the dataset and save it in the output_file (csv format)
        """
        from tqdm import tqdm

        dataloader = torch.utils.data.DataLoader(
            self.source_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.nb_workers,
        )
        with open(self.output_file, "w") as f:
            from csv import writer

            csv_writer = writer(f, delimiter="|")
            # Write the header
            csv_writer.writerow(["cik", "sample_idx", "full_text", "label"])
            for batch in tqdm(
                dataloader, f"Extracting Sftdataset to {self.output_file}"
            ):
                cik = batch["cik"]
                sample_idx = batch["sample_idx"]
                full_text = batch["full_text"]
                label = batch["label"]
                rows = zip(cik, sample_idx, full_text, label)
                csv_writer.writerows(rows)

        print(f"Dataset extracted to {self.output_file}")


if __name__ == "__main__":
    import argparse

    from researchpkg.industry_classification.dataset.sec_transformer_datamodule import (
        TextTransformerTemplateType,
    )

    parser = argparse.ArgumentParser(
        prog="SftDatasetProcessor", description="Run something"
    )

    parser.add_argument(
        "-g",
        "--global_exp_name",
        help="The global experiment name to use. Should match one of existing generated datasets.\
        Example : _count30_sic1agg, _count30_sic1agg_with12ratios",
    )

    parser.add_argument(
        "--template_type",
        default=TextTransformerTemplateType.DESCRIPTIVE,
        help="The template to use for the text transformer",
    )

    parser.add_argument(
        "--max_tag_depth",
        type=int,
        default=None,
        help="The maximum depth of tags in the taxonomy to considerer.By default all tags are considered.",
    )

    args = parser.parse_args()
    template_type = args.template_type
    max_tag_depth = args.max_tag_depth

    global_exp_name = args.global_exp_name
    ExperimentUtils.check_global_experiment_name(global_exp_name)
    dataset_dir = os.path.join(SEC_ROOT_DATA_DIR, f"clean_with_gaap{global_exp_name}")

    depth_text = f"_depth_{max_tag_depth}" if max_tag_depth is not None else ""
    output_dir = os.path.join(dataset_dir, f"sft_dataset_{template_type}{depth_text}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for datasetType in [DatasetType.TRAIN, DatasetType.VAL]:
        source_dataset = SecTextTransformerDataset(
            dataset_dir,
            datasetType,
            tokenizer=None,
            seq_max_length=1024,
            template_type=template_type,
            max_tag_depth=max_tag_depth,
            load_in_memory=False,
            return_raw=True,
        )
        output_file = os.path.join(output_dir, f"{datasettype}.csv")
        processor = SftDatasetProcessor(source_dataset, output_file)
        processor.extract_all()
