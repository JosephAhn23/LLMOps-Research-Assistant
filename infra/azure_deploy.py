"""
Azure deployment — Azure Container Apps + Azure ML.

Provides parity with the existing AWS/SageMaker deployment layer:
  - Azure Container Apps (ACA) for the FastAPI + worker services
  - Azure Container Registry (ACR) for image storage
  - Azure ML workspace for model training and registry
  - Azure Monitor / Application Insights for observability
  - Bicep-compatible ARM resource definitions (via azure-mgmt-*)

Covers gaps: Azure, multi-cloud deployment, Azure ML, ACA microservices.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AzureConfig:
    subscription_id: str = field(
        default_factory=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    )
    resource_group: str = field(
        default_factory=lambda: os.environ.get("AZURE_RESOURCE_GROUP", "llmops-rg")
    )
    location: str = "eastus"

    # Container Apps
    aca_environment: str = "llmops-env"
    api_app_name: str = "llmops-api"
    worker_app_name: str = "llmops-worker"
    api_image: str = "llmopsacr.azurecr.io/llmops-api:latest"
    worker_image: str = "llmopsacr.azurecr.io/llmops-worker:latest"
    api_cpu: float = 1.0
    api_memory: str = "2Gi"
    worker_cpu: float = 2.0
    worker_memory: str = "4Gi"
    min_replicas: int = 1
    max_replicas: int = 10

    # Azure Container Registry
    acr_name: str = "llmopsacr"
    acr_sku: str = "Standard"

    # Azure ML
    aml_workspace: str = "llmops-aml"
    aml_compute_cluster: str = "gpu-cluster"
    aml_vm_size: str = "Standard_NC6s_v3"   # 1x V100
    aml_min_nodes: int = 0
    aml_max_nodes: int = 4

    # Monitoring
    log_analytics_workspace: str = "llmops-logs"
    app_insights_name: str = "llmops-insights"


# ---------------------------------------------------------------------------
# Azure Container Apps deployment
# ---------------------------------------------------------------------------


class AzureContainerAppsDeployer:
    """
    Deploy the LLMOps API and worker as Azure Container Apps.

    Azure Container Apps provides:
      - Serverless container hosting with scale-to-zero
      - Built-in HTTPS ingress + custom domains
      - KEDA-based autoscaling (HTTP, queue depth, CPU/memory)
      - Dapr sidecar support for service-to-service communication
      - Managed identities for secrets (no credentials in env vars)
    """

    def __init__(self, config: AzureConfig):
        self.config = config
        self._credential = None
        self._aca_client = None

    def _get_credential(self):
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential
                self._credential = DefaultAzureCredential()
            except ImportError:
                raise ImportError(
                    "Install azure-identity: pip install azure-identity"
                )
        return self._credential

    def _get_aca_client(self):
        if self._aca_client is None:
            try:
                from azure.mgmt.appcontainers import ContainerAppsAPIClient
                self._aca_client = ContainerAppsAPIClient(
                    credential=self._get_credential(),
                    subscription_id=self.config.subscription_id,
                )
            except ImportError:
                raise ImportError(
                    "Install azure-mgmt-appcontainers: "
                    "pip install azure-mgmt-appcontainers"
                )
        return self._aca_client

    def create_managed_environment(self) -> dict:
        """Create the Container Apps managed environment (shared infrastructure)."""
        client = self._get_aca_client()
        cfg = self.config

        env_def = {
            "location": cfg.location,
            "properties": {
                "appLogsConfiguration": {
                    "destination": "log-analytics",
                    "logAnalyticsConfiguration": {
                        "customerId": cfg.log_analytics_workspace,
                    },
                },
                "daprAIInstrumentationKey": "",
            },
        }

        logger.info("Creating Container Apps environment: %s", cfg.aca_environment)
        poller = client.managed_environments.begin_create_or_update(
            resource_group_name=cfg.resource_group,
            environment_name=cfg.aca_environment,
            environment_envelope=env_def,
        )
        result = poller.result()
        logger.info("Environment ready: %s", result.name)
        return {"environment_id": result.id, "name": result.name}

    def deploy_api(self, openai_api_key_secret_uri: str = "") -> dict:
        """Deploy the FastAPI service as a Container App with HTTP ingress."""
        client = self._get_aca_client()
        cfg = self.config

        app_def = {
            "location": cfg.location,
            "properties": {
                "managedEnvironmentId": f"/subscriptions/{cfg.subscription_id}"
                    f"/resourceGroups/{cfg.resource_group}"
                    f"/providers/Microsoft.App/managedEnvironments/{cfg.aca_environment}",
                "configuration": {
                    "ingress": {
                        "external": True,
                        "targetPort": 8000,
                        "transport": "http",
                        "allowInsecure": False,
                    },
                    "secrets": [
                        {
                            "name": "openai-api-key",
                            "keyVaultUrl": openai_api_key_secret_uri,
                            "identity": "system",
                        }
                    ] if openai_api_key_secret_uri else [],
                    "registries": [
                        {
                            "server": f"{cfg.acr_name}.azurecr.io",
                            "identity": "system",
                        }
                    ],
                },
                "template": {
                    "containers": [
                        {
                            "name": "api",
                            "image": cfg.api_image,
                            "resources": {
                                "cpu": cfg.api_cpu,
                                "memory": cfg.api_memory,
                            },
                            "env": [
                                {"name": "OPENAI_API_KEY", "secretRef": "openai-api-key"}
                                if openai_api_key_secret_uri
                                else {"name": "OPENAI_API_KEY", "value": ""},
                                {"name": "REDIS_URL", "value": "redis://redis:6379/0"},
                            ],
                            "probes": [
                                {
                                    "type": "liveness",
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 30,
                                }
                            ],
                        }
                    ],
                    "scale": {
                        "minReplicas": cfg.min_replicas,
                        "maxReplicas": cfg.max_replicas,
                        "rules": [
                            {
                                "name": "http-scaling",
                                "http": {"metadata": {"concurrentRequests": "50"}},
                            }
                        ],
                    },
                },
            },
        }

        logger.info("Deploying API Container App: %s", cfg.api_app_name)
        poller = client.container_apps.begin_create_or_update(
            resource_group_name=cfg.resource_group,
            container_app_name=cfg.api_app_name,
            container_app_envelope=app_def,
        )
        result = poller.result()
        fqdn = result.properties.configuration.ingress.fqdn if result.properties else ""
        logger.info("API deployed at: https://%s", fqdn)
        return {"app_name": result.name, "fqdn": fqdn, "id": result.id}

    def deploy_worker(self, redis_connection_string: str = "") -> dict:
        """Deploy the Celery worker as a Container App (no ingress, queue-scaled)."""
        client = self._get_aca_client()
        cfg = self.config

        app_def = {
            "location": cfg.location,
            "properties": {
                "managedEnvironmentId": f"/subscriptions/{cfg.subscription_id}"
                    f"/resourceGroups/{cfg.resource_group}"
                    f"/providers/Microsoft.App/managedEnvironments/{cfg.aca_environment}",
                "configuration": {
                    "ingress": None,   # no public ingress for workers
                    "secrets": [
                        {"name": "redis-conn", "value": redis_connection_string}
                    ] if redis_connection_string else [],
                },
                "template": {
                    "containers": [
                        {
                            "name": "worker",
                            "image": cfg.worker_image,
                            "resources": {
                                "cpu": cfg.worker_cpu,
                                "memory": cfg.worker_memory,
                            },
                            "command": [
                                "celery", "-A", "ingestion.batch_processor",
                                "worker", "--loglevel=info",
                            ],
                            "env": [
                                {"name": "REDIS_URL", "secretRef": "redis-conn"}
                                if redis_connection_string
                                else {"name": "REDIS_URL", "value": "redis://localhost:6379/0"},
                            ],
                        }
                    ],
                    "scale": {
                        "minReplicas": 0,   # scale-to-zero when idle
                        "maxReplicas": cfg.max_replicas,
                        "rules": [
                            {
                                "name": "redis-queue-scaling",
                                "custom": {
                                    "type": "redis",
                                    "metadata": {
                                        "listName": "celery",
                                        "listLength": "10",
                                    },
                                    "auth": [
                                        {
                                            "secretRef": "redis-conn",
                                            "triggerParameter": "redisConnectionString",
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                },
            },
        }

        logger.info("Deploying worker Container App: %s", cfg.worker_app_name)
        poller = client.container_apps.begin_create_or_update(
            resource_group_name=cfg.resource_group,
            container_app_name=cfg.worker_app_name,
            container_app_envelope=app_def,
        )
        result = poller.result()
        logger.info("Worker deployed: %s", result.name)
        return {"app_name": result.name, "id": result.id}

    def get_deployment_status(self) -> dict:
        """Return the current status of all deployed Container Apps."""
        client = self._get_aca_client()
        cfg = self.config

        status = {}
        for app_name in [cfg.api_app_name, cfg.worker_app_name]:
            try:
                app = client.container_apps.get(cfg.resource_group, app_name)
                replicas = client.container_apps_revisions.list_replicas(
                    cfg.resource_group, app_name,
                    app.properties.latest_revision_name or "",
                )
                status[app_name] = {
                    "provisioning_state": app.properties.provisioning_state,
                    "latest_revision": app.properties.latest_revision_name,
                    "replicas": [r.name for r in replicas],
                }
            except Exception as exc:
                status[app_name] = {"error": str(exc)}

        return status


# ---------------------------------------------------------------------------
# Azure ML workspace + compute
# ---------------------------------------------------------------------------


class AzureMLDeployer:
    """
    Azure ML workspace management for model training and registry.

    Provides the same capabilities as the SageMaker deployment layer
    (infra/sagemaker_pipeline.py) but targeting Azure ML:
      - Managed compute clusters (GPU-backed)
      - Experiment tracking (maps to MLflow runs)
      - Model registry with approval gates
      - Online endpoints for real-time inference
    """

    def __init__(self, config: AzureConfig):
        self.config = config
        self._ml_client = None

    def _get_ml_client(self):
        if self._ml_client is None:
            try:
                from azure.ai.ml import MLClient
                from azure.identity import DefaultAzureCredential
                self._ml_client = MLClient(
                    credential=DefaultAzureCredential(),
                    subscription_id=self.config.subscription_id,
                    resource_group_name=self.config.resource_group,
                    workspace_name=self.config.aml_workspace,
                )
            except ImportError:
                raise ImportError(
                    "Install azure-ai-ml: pip install azure-ai-ml azure-identity"
                )
        return self._ml_client

    def create_compute_cluster(self) -> dict:
        """Create a GPU compute cluster for distributed training."""
        from azure.ai.ml.entities import AmlCompute

        client = self._get_ml_client()
        cfg = self.config

        cluster = AmlCompute(
            name=cfg.aml_compute_cluster,
            type="amlcompute",
            size=cfg.aml_vm_size,
            min_instances=cfg.aml_min_nodes,
            max_instances=cfg.aml_max_nodes,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        result = client.compute.begin_create_or_update(cluster).result()
        logger.info(
            "Compute cluster ready: %s (%s, max=%d)",
            result.name, cfg.aml_vm_size, cfg.aml_max_nodes,
        )
        return {"name": result.name, "vm_size": cfg.aml_vm_size}

    def submit_finetune_job(
        self,
        script_path: str = "finetune/peft_lora_finetune.py",
        experiment_name: str = "qlora-finetune",
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        num_epochs: int = 3,
    ) -> dict:
        """Submit a QLoRA fine-tuning job to the Azure ML compute cluster."""
        from azure.ai.ml import command
        from azure.ai.ml.entities import Environment

        client = self._get_ml_client()
        cfg = self.config

        env = Environment(
            name="llmops-finetune-env",
            image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest",
            conda_file={
                "name": "llmops",
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.11",
                    {"pip": [
                        "transformers==4.39.0",
                        "peft==0.10.0",
                        "accelerate==0.27.2",
                        "bitsandbytes==0.43.0",
                        "datasets==2.18.0",
                        "mlflow==2.11.1",
                    ]},
                ],
            },
        )

        job = command(
            code=".",
            command=(
                f"python {script_path} "
                f"--base-model {base_model} "
                f"--num-epochs {num_epochs}"
            ),
            environment=env,
            compute=cfg.aml_compute_cluster,
            experiment_name=experiment_name,
            display_name=f"qlora-{base_model.split('/')[-1]}-{num_epochs}ep",
        )

        submitted = client.jobs.create_or_update(job)
        logger.info("Job submitted: %s", submitted.name)
        return {
            "job_name": submitted.name,
            "status": submitted.status,
            "studio_url": submitted.studio_url,
        }

    def register_model(
        self,
        model_path: str,
        model_name: str = "llmops-rag-model",
        description: str = "QLoRA fine-tuned RAG model",
        tags: Optional[dict] = None,
    ) -> dict:
        """Register a model in the Azure ML model registry."""
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes

        client = self._get_ml_client()
        model = Model(
            path=model_path,
            name=model_name,
            description=description,
            type=AssetTypes.CUSTOM_MODEL,
            tags=tags or {},
        )
        registered = client.models.create_or_update(model)
        logger.info(
            "Model registered: %s v%s", registered.name, registered.version
        )
        return {
            "name": registered.name,
            "version": registered.version,
            "id": registered.id,
        }

    def deploy_online_endpoint(
        self,
        endpoint_name: str = "llmops-endpoint",
        model_name: str = "llmops-rag-model",
        model_version: str = "1",
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
    ) -> dict:
        """Deploy a model to an Azure ML online endpoint for real-time inference."""
        from azure.ai.ml.entities import (
            ManagedOnlineDeployment,
            ManagedOnlineEndpoint,
        )

        client = self._get_ml_client()

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key",
            tags={"project": "llmops-research-assistant"},
        )
        client.online_endpoints.begin_create_or_update(endpoint).result()

        deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=f"azureml:{model_name}:{model_version}",
            instance_type=instance_type,
            instance_count=instance_count,
            liveness_probe={"initial_delay": 30, "period": 30, "timeout": 10},
            readiness_probe={"initial_delay": 10, "period": 10, "timeout": 5},
        )
        client.online_deployments.begin_create_or_update(deployment).result()

        endpoint_obj = client.online_endpoints.get(endpoint_name)
        scoring_uri = endpoint_obj.scoring_uri
        logger.info("Endpoint deployed: %s", scoring_uri)
        return {
            "endpoint_name": endpoint_name,
            "scoring_uri": scoring_uri,
            "deployment": "blue",
        }


# ---------------------------------------------------------------------------
# Azure Monitor / Application Insights
# ---------------------------------------------------------------------------


class AzureMonitorObservability:
    """
    Azure Monitor integration — mirrors infra/aws_observability.py for Azure.

    Emits custom metrics and structured logs to Application Insights,
    enabling the same observability coverage as the CloudWatch layer.
    """

    def __init__(self, connection_string: str = ""):
        self.connection_string = connection_string or os.environ.get(
            "APPLICATIONINSIGHTS_CONNECTION_STRING", ""
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                provider = TracerProvider()
                exporter = AzureMonitorTraceExporter(
                    connection_string=self.connection_string
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(provider)
                self._client = trace.get_tracer(__name__)
            except ImportError:
                logger.warning(
                    "azure-monitor-opentelemetry-exporter not installed — "
                    "metrics will be logged locally only"
                )
                self._client = None
        return self._client

    def emit_metric(self, name: str, value: float, properties: dict = None):
        """Emit a custom metric to Application Insights."""
        logger.info("METRIC %s=%.4f props=%s", name, value, properties or {})
        tracer = self._get_client()
        if tracer is None:
            return
        with tracer.start_as_current_span(name) as span:
            span.set_attribute("metric.value", value)
            for k, v in (properties or {}).items():
                span.set_attribute(k, str(v))

    def track_rag_query(
        self,
        query: str,
        latency_ms: float,
        sources_retrieved: int,
        faithfulness: float = 0.0,
    ):
        self.emit_metric("rag.query.latency_ms", latency_ms, {"query_len": len(query)})
        self.emit_metric("rag.sources_retrieved", float(sources_retrieved))
        if faithfulness:
            self.emit_metric("rag.faithfulness", faithfulness)

    def track_model_inference(
        self,
        model: str,
        tokens_generated: int,
        latency_ms: float,
    ):
        self.emit_metric(
            "model.inference.latency_ms",
            latency_ms,
            {"model": model, "tokens": tokens_generated},
        )
        if latency_ms > 0:
            self.emit_metric(
                "model.tokens_per_second",
                tokens_generated / (latency_ms / 1000),
                {"model": model},
            )


# ---------------------------------------------------------------------------
# Bicep template generator (IaC)
# ---------------------------------------------------------------------------


def generate_bicep_template(config: AzureConfig) -> str:
    """
    Generate a Bicep template for the full Azure deployment.

    Bicep is the preferred Azure IaC language (compiles to ARM JSON).
    This provides the same IaC coverage as infra/terraform/main.tf for AWS.
    """
    return f"""// LLMOps Research Assistant — Azure Infrastructure
// Generated by infra/azure_deploy.py
// Deploy: az deployment group create --resource-group {config.resource_group} --template-file main.bicep

param location string = '{config.location}'
param acrName string = '{config.acr_name}'
param acaEnvironmentName string = '{config.aca_environment}'
param apiAppName string = '{config.api_app_name}'
param workerAppName string = '{config.worker_app_name}'
param amlWorkspaceName string = '{config.aml_workspace}'
param logAnalyticsName string = '{config.log_analytics_workspace}'
param appInsightsName string = '{config.app_insights_name}'

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {{
  name: logAnalyticsName
  location: location
  properties: {{
    sku: {{ name: 'PerGB2018' }}
    retentionInDays: 30
  }}
}}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {{
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {{
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }}
}}

// Azure Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {{
  name: acrName
  location: location
  sku: {{ name: '{config.acr_sku}' }}
  properties: {{
    adminUserEnabled: false
    anonymousPullEnabled: false
  }}
}}

// Container Apps Managed Environment
resource acaEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {{
  name: acaEnvironmentName
  location: location
  properties: {{
    appLogsConfiguration: {{
      destination: 'log-analytics'
      logAnalyticsConfiguration: {{
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }}
    }}
    daprAIInstrumentationKey: appInsights.properties.InstrumentationKey
  }}
}}

// API Container App
resource apiApp 'Microsoft.App/containerApps@2023-05-01' = {{
  name: apiAppName
  location: location
  identity: {{ type: 'SystemAssigned' }}
  properties: {{
    managedEnvironmentId: acaEnv.id
    configuration: {{
      ingress: {{
        external: true
        targetPort: 8000
        transport: 'http'
      }}
      registries: [{{
        server: acr.properties.loginServer
        identity: 'system'
      }}]
    }}
    template: {{
      containers: [{{
        name: 'api'
        image: '${{acr.properties.loginServer}}/llmops-api:latest'
        resources: {{
          cpu: json('{config.api_cpu}')
          memory: '{config.api_memory}'
        }}
        probes: [{{
          type: 'Liveness'
          httpGet: {{ path: '/health' port: 8000 }}
          initialDelaySeconds: 10
          periodSeconds: 30
        }}]
      }}]
      scale: {{
        minReplicas: {config.min_replicas}
        maxReplicas: {config.max_replicas}
        rules: [{{
          name: 'http-scaling'
          http: {{ metadata: {{ concurrentRequests: '50' }} }}
        }}]
      }}
    }}
  }}
}}

// Worker Container App (scale-to-zero, Redis KEDA trigger)
resource workerApp 'Microsoft.App/containerApps@2023-05-01' = {{
  name: workerAppName
  location: location
  identity: {{ type: 'SystemAssigned' }}
  properties: {{
    managedEnvironmentId: acaEnv.id
    configuration: {{
      registries: [{{
        server: acr.properties.loginServer
        identity: 'system'
      }}]
    }}
    template: {{
      containers: [{{
        name: 'worker'
        image: '${{acr.properties.loginServer}}/llmops-worker:latest'
        resources: {{
          cpu: json('{config.worker_cpu}')
          memory: '{config.worker_memory}'
        }}
        command: ['celery', '-A', 'ingestion.batch_processor', 'worker', '--loglevel=info']
      }}]
      scale: {{
        minReplicas: 0
        maxReplicas: {config.max_replicas}
      }}
    }}
  }}
}}

// Azure ML Workspace
resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-06-01-preview' = {{
  name: amlWorkspaceName
  location: location
  identity: {{ type: 'SystemAssigned' }}
  properties: {{
    applicationInsights: appInsights.id
    containerRegistry: acr.id
  }}
}}

output apiUrl string = 'https://${{apiApp.properties.configuration.ingress.fqdn}}'
output acrLoginServer string = acr.properties.loginServer
output amlWorkspaceId string = amlWorkspace.id
"""


def save_bicep_template(config: AzureConfig, output_path: str = "infra/azure/main.bicep"):
    """Write the Bicep template to disk."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generate_bicep_template(config))
    logger.info("Bicep template written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure deployment utilities")
    sub = parser.add_subparsers(dest="command")

    gen_bicep = sub.add_parser("gen-bicep", help="Generate Bicep IaC template")
    gen_bicep.add_argument("--output", default="infra/azure/main.bicep")

    deploy_cmd = sub.add_parser("deploy", help="Deploy to Azure Container Apps")
    deploy_cmd.add_argument("--resource-group", default="llmops-rg")
    deploy_cmd.add_argument("--location", default="eastus")

    args = parser.parse_args()
    cfg = AzureConfig(
        resource_group=getattr(args, "resource_group", "llmops-rg"),
        location=getattr(args, "location", "eastus"),
    )

    if args.command == "gen-bicep":
        path = save_bicep_template(cfg, args.output)
        print(f"Bicep template written to: {path}")
    elif args.command == "deploy":
        deployer = AzureContainerAppsDeployer(cfg)
        env = deployer.create_managed_environment()
        api = deployer.deploy_api()
        worker = deployer.deploy_worker()
        print(json.dumps({"environment": env, "api": api, "worker": worker}, indent=2))
    else:
        parser.print_help()
