"""
SageMaker Pipeline for automated retraining - not just deployment.
Covers: SageMaker (real usage, not just deploy script)
"""
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.parameters import ParameterString

role = sagemaker.get_execution_role()
session = sagemaker.Session()
region = boto3.Session().region_name

input_data = ParameterString(name="InputData", default_value="s3://your-bucket/data/")
model_output = ParameterString(name="ModelOutput", default_value="s3://your-bucket/models/")


def build_pipeline() -> Pipeline:
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
    )
    processing_step = ProcessingStep(
        name="PrepareData",
        processor=processor,
        code="infra/sm_processing.py",
        inputs=[sagemaker.processing.ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/output")],
    )

    huggingface_estimator = HuggingFace(
        entry_point="finetune/lora_finetune.py",
        role=role,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters={
            "epochs": 3,
            "lora_rank": 8,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        environment={"MLFLOW_TRACKING_URI": "your-mlflow-server"},
    )
    training_step = TrainingStep(
        name="LoRAFinetune",
        estimator=huggingface_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            )
        },
    )

    pipeline = Pipeline(
        name="LLMOpsRetrainingPipeline",
        parameters=[input_data, model_output],
        steps=[processing_step, training_step],
        sagemaker_session=session,
    )
    return pipeline


if __name__ == "__main__":
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Pipeline started: {execution.arn}")
