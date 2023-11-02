import boto3
import logging
import sagemaker
from sagemaker import estimator, LocalSession
from sagemaker.pytorch import PyTorch
from dataclasses import dataclass

print(sagemaker.__version__)
print(boto3.__version__)

sagemaker_logger = logging.getLogger('sagemaker')
sagemaker_logger.setLevel(logging.DEBUG)


@dataclass
class AWSSagemakerConfig:
    role: str = 'arn:aws:iam::688364882631:role/SagemakerEstimator'
    local: bool = False
    max_run: int = 129600  # Set timeout to 36 hours (default is 86400 seconds, or 24 hours)

    local_training_input_path = 'file://timf3\PycharmProjects\AFL-Data\marvel\\afl-preprocessed'

    # Path to directory containing training data
    s3_uri_training_data = 's3://dublin-afl-preprocessed/'

    # S3 URI for model artifacts
    # (non weights, weights are saved to the checkpoints folder)
    s3_model_artifacts = 's3://afl-training-artifacts/'
    local_model_artifacts_output_path = 'file://model/'

    # S3 URI for weights (not an environment variable needs to be set manually) - opt/ml/checkpoints
    if not local:
        # Our checkpoints folder.
        checkpoint_local_path = '/opt/ml/checkpoints'
        checkpoint_s3_uri = 's3://afl-training-artifacts/checkpoints/'

    # Local files.
    entry_point: str = 'train.py'
    source_dir: str = r'C:\Users\timf3\PycharmProjects\BallNet'

    # Estimator config
    framework_version: str = '1.12.1'
    train_instance_count: int = 1
    py_version: str = 'py38'

    # This allows us to stream data from S3 to the training instance, rather than having to download the entire dataset - It should be faster.
    file_input_mode: str = "FastFile"

    # Instance types page (incl. info on cost): https://www.notion.so/Sagemaker-Estimator-63193c013110465989a520706d462565
    # Cheap for testing: ml.m5.large
    # Better performance for training: ml.p3.2xlarge or ml.p2.xlarge
    train_instance_type: str = 'local' if local else 'ml.p2.xlarge'


def initialise_session(config: AWSSagemakerConfig) -> sagemaker.Session:
    """
    This function initialises the session
    """
    sess = LocalSession() if config.local else sagemaker.Session(boto3.session.Session(region_name='eu-west-1'))
    print(f"Session: {sess}")
    print(f"Session role: {config.role}")
    print(f"Session region: {sess.boto_session.region_name}")

    return sess


def initialise_estimator(config: AWSSagemakerConfig, sess: sagemaker.Session) -> estimator.Estimator:
    """
    This function initialises the estimator
    Docs for PyTorch estimator: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
    """

    print("Initialising estimator...")

    # Requirements
    env = {
        'SAGEMAKER_REQUIREMENTS': 'requirements.txt',  # path relative to `source_dir` below.
    }

    if config.local:
        estimator = PyTorch(
            entry_point=config.entry_point,
            py_version=config.py_version,
            source_dir=config.source_dir,
            role=config.role,
            env=env,
            framework_version=config.framework_version,
            instance_count=config.train_instance_count,
            instance_type=config.train_instance_type,
            output_path=config.local_model_artifacts_output_path,
            sagemaker_session=sess
        )
    else:
        # Initialise the estimator
        estimator = PyTorch(
            entry_point=config.entry_point,
            py_version=config.py_version,
            source_dir=config.source_dir,
            role=config.role,
            env=env,
            input_mode="FastFile",
            framework_version=config.framework_version,
            instance_count=config.train_instance_count,
            instance_type=config.train_instance_type,
            output_path=config.s3_model_artifacts,
            sagemaker_session=sess,
            checkpoint_s3_uri=config.checkpoint_s3_uri,
            checkpoint_local_path=config.checkpoint_local_path,
            max_run=config.max_run
        )

    print("Estimator initialised.")

    return estimator


def run_estimator() -> None:
    """
    This function runs the estimator on the training data
    """
    config = AWSSagemakerConfig()
    sess = initialise_session(config)
    estimator = initialise_estimator(config, sess)
    sagemaker.inputs.TrainingInput.input_mode = "FastFile"

    # Train the estimator
    estimator.fit({'training': config.s3_uri_training_data})


if __name__ == "__main__":
    run_estimator()
