"""
SageMaker Model Registry + A/B deployment + approval gate.
Covers: SageMaker weakness - not just a pipeline, actual model management
"""
import os
import time
from datetime import datetime


MODEL_PACKAGE_GROUP = os.getenv("SAGEMAKER_MODEL_GROUP", "llmops-embedder")


class SageMakerModelRegistry:
    """
    Full model lifecycle management:
    - Register models with metrics
    - Approval gates (manual + automated)
    - A/B traffic splitting
    - Rollback on metric degradation

    All AWS clients and the SageMaker execution role are lazy-initialized so
    that importing this module outside a SageMaker environment (local dev, CI,
    tests) does not crash at import time.
    """

    def __init__(self, region: str | None = None):
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._s3_bucket = os.getenv("S3_BUCKET", "llmops-research-assistant")
        self._role: str | None = None
        self._sm_client = None
        self._session = None

    # ------------------------------------------------------------------
    # Lazy properties — nothing is imported or called until first use
    # ------------------------------------------------------------------

    @property
    def role(self) -> str:
        if self._role is None:
            import sagemaker
            self._role = sagemaker.get_execution_role()
        return self._role

    @property
    def sm_client(self):
        if self._sm_client is None:
            import boto3
            self._sm_client = boto3.client("sagemaker", region_name=self._region)
        return self._sm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_artifact_s3: str,
        evaluation_metrics: dict,
        model_image_uri: str,
        description: str = "",
    ) -> str:
        """Register a trained model in SageMaker Model Registry."""
        try:
            self.sm_client.create_model_package_group(
                ModelPackageGroupName=MODEL_PACKAGE_GROUP,
                ModelPackageGroupDescription="LLMOps embedding model versions",
            )
        except Exception as exc:
            # Ignore "already exists"; re-raise quota / permission errors.
            if "already exists" not in str(exc).lower():
                raise

        model_package = self.sm_client.create_model_package(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageDescription=description or f"LLMOps embedder {datetime.now().isoformat()}",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": model_image_uri,
                        "ModelDataUrl": model_artifact_s3,
                        "Framework": "PYTORCH",
                        "FrameworkVersion": "2.1",
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.g4dn.xlarge", "ml.m5.xlarge"],
            },
            ModelApprovalStatus="PendingManualApproval",
            ModelMetrics={
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": model_artifact_s3 + "/metrics.json",
                    }
                }
            },
            CustomerMetadataProperties={
                "faithfulness": str(evaluation_metrics.get("faithfulness", 0)),
                "answer_relevancy": str(evaluation_metrics.get("answer_relevancy", 0)),
                "mean_ragas_score": str(evaluation_metrics.get("mean_score", 0)),
            },
        )

        model_package_arn = model_package["ModelPackageArn"]
        print(f"Registered model: {model_package_arn}")
        print("Status: PendingManualApproval - awaiting human sign-off")
        return model_package_arn

    def auto_approve_if_metrics_pass(
        self,
        model_package_arn: str,
        min_faithfulness: float = 0.85,
        min_mean_score: float = 0.80,
    ) -> bool:
        """
        Automated approval gate - approves only if metrics exceed thresholds.
        Covers: Conditional deployment based on quality metrics.
        """
        response = self.sm_client.describe_model_package(ModelPackageName=model_package_arn)
        metadata = response.get("CustomerMetadataProperties", {})

        faithfulness = float(metadata.get("faithfulness", 0))
        mean_score = float(metadata.get("mean_ragas_score", 0))

        passes_gate = faithfulness >= min_faithfulness and mean_score >= min_mean_score

        if passes_gate:
            self.sm_client.update_model_package(
                ModelPackageName=model_package_arn,
                ModelApprovalStatus="Approved",
                ApprovalDescription=f"Auto-approved: faithfulness={faithfulness:.3f}, mean={mean_score:.3f}",
            )
            print(f"Auto-approved: faithfulness={faithfulness:.3f} >= {min_faithfulness}")
        else:
            self.sm_client.update_model_package(
                ModelPackageName=model_package_arn,
                ModelApprovalStatus="Rejected",
                ApprovalDescription=(
                    f"Rejected: faithfulness={faithfulness:.3f} < {min_faithfulness} "
                    f"or mean={mean_score:.3f} < {min_mean_score}"
                ),
            )
            print("Rejected: metrics below threshold")

        return passes_gate

    def deploy_ab_test(
        self,
        endpoint_name: str,
        production_model_arn: str,
        candidate_model_arn: str,
        candidate_traffic_pct: int = 10,
    ) -> str:
        """
        Deploy A/B test: 90% traffic to production, 10% to candidate.
        Covers: A/B deployment - the SageMaker weakness
        """
        production_variant = {
            "VariantName": "production",
            "ModelName": self._get_model_name(production_model_arn),
            "InstanceType": "ml.g4dn.xlarge",
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 100 - candidate_traffic_pct,
        }

        candidate_variant = {
            "VariantName": "candidate",
            "ModelName": self._get_model_name(candidate_model_arn),
            "InstanceType": "ml.g4dn.xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": candidate_traffic_pct,
        }

        data_capture_uri = (
            f"s3://{self._s3_bucket}/data-capture/{endpoint_name}"
        )

        try:
            self.sm_client.create_endpoint_config(
                EndpointConfigName=f"{endpoint_name}-ab-config",
                ProductionVariants=[production_variant, candidate_variant],
                DataCaptureConfig={
                    "EnableCapture": True,
                    "InitialSamplingPercentage": 100,
                    "DestinationS3Uri": data_capture_uri,
                    "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                },
            )

            self.sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=f"{endpoint_name}-ab-config",
            )
            print(f"A/B endpoint deploying: {endpoint_name}")
            print(f"  Production: {100 - candidate_traffic_pct}% traffic")
            print(f"  Candidate:  {candidate_traffic_pct}% traffic")

        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
            self.sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=f"{endpoint_name}-ab-config",
            )

        self._wait_for_endpoint(endpoint_name)
        return endpoint_name

    def promote_candidate(self, endpoint_name: str):
        """Promote candidate to 100% traffic after successful A/B test."""
        response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = response["EndpointConfigName"]
        config = self.sm_client.describe_endpoint_config(EndpointConfigName=config_name)

        candidate = next(v for v in config["ProductionVariants"] if v["VariantName"] == "candidate")

        new_config_name = f"{endpoint_name}-promoted-{int(time.time())}"
        self.sm_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=[
                {
                    **candidate,
                    "VariantName": "production",
                    "InitialVariantWeight": 100,
                }
            ],
        )

        self.sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name,
        )
        print(f"Candidate promoted to 100% traffic on {endpoint_name}")
        self._wait_for_endpoint(endpoint_name)

    def rollback(self, endpoint_name: str, previous_config_name: str):
        """Rollback to previous endpoint config on degradation."""
        self.sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=previous_config_name,
        )
        print(f"Rolled back {endpoint_name} to {previous_config_name}")
        self._wait_for_endpoint(endpoint_name)

    def _get_model_name(self, model_package_arn: str) -> str:
        """Create deployable SageMaker model from package ARN."""
        model_name = f"llmops-model-{int(time.time())}"
        self.sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "ModelPackageName": model_package_arn,
            },
            ExecutionRoleArn=self.role,
        )
        return model_name

    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 600):
        """Poll until endpoint is InService."""
        print(f"Waiting for endpoint {endpoint_name}...")
        start = time.time()
        while time.time() - start < timeout:
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]
            if status == "InService":
                print(f"Endpoint {endpoint_name} is InService")
                return
            if status in ["Failed", "RollingBack"]:
                raise RuntimeError(f"Endpoint {endpoint_name} entered {status}")
            time.sleep(15)
        raise TimeoutError(f"Endpoint {endpoint_name} did not become InService in {timeout}s")

    def list_model_versions(self) -> list:
        """List all registered model versions with approval status."""
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            SortBy="CreationTime",
            SortOrder="Descending",
        )
        versions = []
        for pkg in response["ModelPackageSummaryList"]:
            versions.append(
                {
                    "arn": pkg["ModelPackageArn"],
                    "version": pkg["ModelPackageVersion"],
                    "status": pkg["ModelApprovalStatus"],
                    "created": pkg["CreationTime"].isoformat(),
                }
            )
        return versions
