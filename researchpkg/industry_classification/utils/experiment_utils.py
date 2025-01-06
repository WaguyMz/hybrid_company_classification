import os
import pyaml
from yaml import Loader

from researchpkg.industry_classification.config import GLOBAL_EXP_INDEX, ROOT_DIR

class ExperimentUtils:
    @staticmethod
    def check_global_experiment_name(global_experiment_name):
        """
        Check if a global experiment exists
        :param global_experiment_name: name of the global experiment
        """
        if global_experiment_name in GLOBAL_EXP_INDEX:
            return True
        else:
            print("Allowed values for -g parameter: \n -----------------------------")

            for exp, desc in GLOBAL_EXP_INDEX.items():
                print(f"{exp}: {desc}")
            print("\n")
            raise ValueError(f"Global experiment {global_experiment_name} not found.")

    @staticmethod
    def check_experiment(experiment_dir):
        """
        Check if an experiment dir exists and contains an experiment.yaml file.
        :param experiment_dir: path to the experiment dir
        """
        return os.path.exists(experiment_dir) and os.path.exists(
            os.path.join(experiment_dir, "experiment.yaml")
        )

    @staticmethod
    def load_experiment_data(experiment_dir):
        exp_file = os.path.join(experiment_dir, "experiment.yaml")
        exp = pyaml.yaml.load(open(exp_file, "r"), Loader=Loader)

        return exp

    @staticmethod
    def save_experiment_data(experiment_dir, exp):
        exp_file = os.path.join(experiment_dir, "experiment.yaml")
        with open(exp_file, "w") as f:
            pyaml.yaml.dump(exp, f)

    @staticmethod
    def initialize_experiment(
        experiment_dir, dataset_dir, model_config, training_config
    ):
        """
        Create experiment dir.
        :param experiment_dir: path to the experiment dir
        :param dataset_dir: path to the dataset dir
        :param model_config: model configuration dict
        :param training_config: training configuration dict
        """
        experiment_name = os.path.basename(experiment_dir)
        os.makedirs(experiment_dir, exist_ok=True)

        dataset_config_file = os.path.join(dataset_dir, "dataset_config.yaml")

        exp_dict = {
            "experiment_name": experiment_name,
            "dataset_config": pyaml.yaml.load(
                open(dataset_config_file, "r"), Loader=Loader
            ),
            "model_config": dict(model_config),
            "training_config": dict(training_config),
        }

        # Save the experiment.yaml file
        ExperimentUtils.save_experiment_data(experiment_dir, exp_dict)

    @staticmethod
    def uptate_experiment_best_model(
        experiment_dir, metric_name, best_model_score, best_epoch, best_model_path
    ):
        """
        Update the yaml file of an experiment
        :param experiment_dir: path to the experiment dir
        :paramm metric_name: name of the metric
        :param best_model_score: best score of the metric
        :param best_epoch: best epoch
        :param best_model_path: path to the best model
        """
        exp = ExperimentUtils.load_experiment_data(experiment_dir)
        exp["best_model"] = {}
        exp["best_model"]["epoch"] = best_epoch
        exp["best_model"]["metric"] = metric_name
        exp["best_model"][metric_name] = round(best_model_score, 5)

        best_model_path = (
            os.path.relpath(best_model_path, ROOT_DIR)
            if best_model_path is not None
            else None
        )
        exp["best_model"]["path"] = best_model_path

        ExperimentUtils.save_experiment_data(experiment_dir, exp)

        # Commit the experiment
        # ExperimentUtils.commit_experiment(experiment_dir)

    @staticmethod
    def get_best_model(experiment_name, experiments_dir):
        """
        Return the best model of an experiment
        :param experiment_name: name of the experiment
        :return dict of {"metric": best_metric, "epoch": best_epoch,
         "path": best_model_path}
        """
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        exp = ExperimentUtils.load_experiment_data(experiment_dir)
        return exp.get("best_model", None)

    @staticmethod
    def update_experiment_results(experiment_dir, metrics_dict):
        """
        Update the results of the experiment results
        :param experiment_dir: path to the experiment dir
        :param metrics_dict: dict of metrics
        """
        exp = ExperimentUtils.load_experiment_data(experiment_dir)
        # Update the experiment results file using the metrics_dict
        exp["results"] = {}
        for k, v in metrics_dict.items():
            exp["results"][k] = v
        ExperimentUtils.save_experiment_data(experiment_dir, exp)

    @staticmethod
    def commit_experiment(experiment_dir):
        """
        Commit an experiment to git:
        Commit the experiment.yaml file and the best model if it exists.

        The following files are commited.
            -experiment.yaml
            -best_model_path if it exists
        :param experiment_dir: path to the experiment dir
        """
        # TODO: Improve this function
        # 1. Check the state of the git repo (No conflict,ect)
        # 2. Check the commit email.

        # from git import Reporepo = Repo(os.path.join(ROOT_DIR, '..')

        # 2. Load the experiment data
        exp = ExperimentUtils.load_experiment_data(experiment_dir)

        # 3. Check if the best_model_path exists
        if "best_model" in exp:
            # TODO : Check the status fo the git repo. Or use GitPython

            # 1. Add the experiment.yaml file to git
            exp_file = os.path.join(experiment_dir, "experiment.yaml")
            add_1 = f'git add "{exp_file}"'
            # Run git add synchronously
            best_model_score = exp["best_model"]["score"]
            metric = exp["best_model"]["metric"]

            # if os.path.exists(best_model_path):
            #     add_2 = f'git add "{best_model_path}"'

            commit_message = (
                f"[BEST_MODEL_COMMIT] {exp['experiment_name']}"
                f" - {metric}={best_model_score}"
            )
            commit_name = os.popen("echo $GIT_COMMITTER_NAME").read().strip()
            commit_email = os.popen("echo $GIT_COMMITTER_EMAIL").read().strip()
            commit_command = (
                f'git commit -m "{commit_message}"'
                f' --author="{commit_name} {commit_email}"'
            )
            current_branch = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
            push_command = f"git push origin {current_branch}"

            os.system(f"{add_1} ; {commit_command} ; {push_command}")

        else:
            print("No best model found. Cannot commit the experiment.")
