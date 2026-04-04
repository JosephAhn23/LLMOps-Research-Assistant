"""
Azure Deployment Module for LLMOps Research Assistant.

Mirrors the existing AWS layer — adds Azure Container Apps + Azure ML + Azure OpenAI.
Covers: Azure Container Apps, Azure ML, Azure OpenAI, Key Vault, Azure Monitor.

Components:
  - AzureConfig               — unified config dataclass (env-var backed)
  - AzureMLManager            — model registry, managed online endpoints, A/B traffic splits
  - AzureContainerAppsDeployer — serverless FastAPI deployment on Azure Container Apps
  - AzureOpenAIBackend        — drop-in replacement for OpenAI client (chat + embeddings + streaming)
  - AzureMonitor              — custom metric emission via Azure Monitor REST API

Mirrors: infra/sagemaker_model_registry.py and infra/aws_observability.py

Requirements (optional — install for Azure deployment):
    azure-identity>=1.16.0
    azure-ai-ml>=1.15.0
    azure-mgmt-appcontainers>=3.0.0
    openai>=1.14.0
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AzureConfig:
    subscription_id: str = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    resource_group: str = os.getenv("AZURE_RESOURCE_GROUP", "llmops-rg")
    location: str = os.getenv("AZURE_LOCATION", "eastus")
    workspace_name: str = os.getenv("AZURE_ML_WORKSPACE", "llmops-aml")
    container_app_name: str = "llmops-api"
    container_app_env: str = "llmops-env"
    acr_name: str = os.getenv("AZURE_ACR_NAME", "llmopsacr")
    key_vault_name: str = os.getenv("AZURE_KEY_VAULT", "llmops-kv")
    openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    openai_api_key: str = os.getenv("AZURE_OPENAI_KEY", "")
    openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Azure ML Manager
# ---------------------------------------------------------------------------

class AzureMLManager:
    """
    Register models, create managed online endpoints, and configure A/B traffic splits.
    Mirrors infra/sagemaker_model_registry.py for the Azure deployment path.
    """

    def __init__(self, config: AzureConfig):
        self.config = config
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            self._client = MLClient(
                DefaultAzureCredential(),
                self.config.subscription_id,
                self.config.resource_group,
                self.config.workspace_name,
            )
            logger.info("Connected to Azure ML workspace: %s", self.config.workspace_name)
        except ImportError:
            logger.warning("azure-ai-ml not installed — using mock client")
            self._client = _MockMLClient()
        except Exception as e:
            logger.warning("Azure ML client init failed: %s — using mock", e)
            self._client = _MockMLClient()

    def register_model(
        self,
        model_path: str,
        model_name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a model in the Azure ML registry. Returns the assigned version string."""
        try:
            from azure.ai.ml.entities import Model
            from azure.ai.ml.constants import AssetTypes
            model = Model(
                path=model_path,
                name=model_name,
                description=description,
                tags=tags or {},
                type=AssetTypes.CUSTOM_MODEL,
            )
            registered = self._client.models.create_or_update(model)
            logger.info("Registered model: %s v%s", model_name, registered.version)
            return registered.version
        except Exception as e:
            logger.error("Model registration failed: %s", e)
            raise

    def deploy_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
        traffic_percent: int = 100,
    ) -> None:
        """Create a managed online endpoint and deploy a model to it."""
        try:
            from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                auth_mode="key",
                tags={"project": "llmops-research-assistant"},
            )
            self._client.online_endpoints.begin_create_or_update(endpoint).wait()

            deployment = ManagedOnlineDeployment(
                name="blue",
                endpoint_name=endpoint_name,
                model=f"azureml:{model_name}:{model_version}",
                instance_type=instance_type,
                instance_count=instance_count,
            )
            self._client.online_deployments.begin_create_or_update(deployment).wait()

            endpoint.traffic = {"blue": traffic_percent}
            self._client.online_endpoints.begin_create_or_update(endpoint).wait()
            logger.info("Endpoint %s live with %d%% traffic", endpoint_name, traffic_percent)
        except Exception as e:
            logger.error("Endpoint deployment failed: %s", e)
            raise

    def ab_deploy(
        self,
        endpoint_name: str,
        blue_model: tuple,   # (model_name, model_version)
        green_model: tuple,  # (model_name, model_version)
        blue_traffic: int = 80,
    ) -> None:
        """
        A/B deployment — two model versions with split traffic.
        Creates both deployments atomically then sets the traffic split.
        """
        try:
            from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint

            for color, (name, version), traffic in [
                ("blue", blue_model, blue_traffic),
                ("green", green_model, 100 - blue_traffic),
            ]:
                dep = ManagedOnlineDeployment(
                    name=color,
                    endpoint_name=endpoint_name,
                    model=f"azureml:{name}:{version}",
                    instance_type="Standard_DS3_v2",
                    instance_count=1,
                )
                self._client.online_deployments.begin_create_or_update(dep).wait()

            endpoint = self._client.online_endpoints.get(endpoint_name)
            endpoint.traffic = {"blue": blue_traffic, "green": 100 - blue_traffic}
            self._client.online_endpoints.begin_create_or_update(endpoint).wait()
            logger.info("A/B split: blue=%d%%, green=%d%%", blue_traffic, 100 - blue_traffic)
        except Exception as e:
            logger.error("A/B deploy failed: %s", e)
            raise

    def list_models(self) -> List[Dict]:
        """List all registered models in the workspace."""
        try:
            return [
                {"name": m.name, "version": m.version, "description": m.description}
                for m in self._client.models.list()
            ]
        except Exception as e:
            logger.error("List models failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# Azure Container Apps Deployer
# ---------------------------------------------------------------------------

class AzureContainerAppsDeployer:
    """
    Deploy the FastAPI gateway to Azure Container Apps (serverless).
    Equivalent to the existing Kubernetes manifests in infra/k8s/.
    """

    def __init__(self, config: AzureConfig):
        self.config = config
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from azure.mgmt.appcontainers import ContainerAppsAPIClient
            from azure.identity import DefaultAzureCredential
            self._client = ContainerAppsAPIClient(
                DefaultAzureCredential(), self.config.subscription_id
            )
        except ImportError:
            logger.warning("azure-mgmt-appcontainers not installed — using mock client")
            self._client = _MockACAClient()
        except Exception as e:
            logger.warning("ACA client init failed: %s — using mock", e)
            self._client = _MockACAClient()

    def _get_env_id(self) -> str:
        env = self._client.managed_environments.get(
            self.config.resource_group, self.config.container_app_env
        )
        return env.id

    def deploy(
        self,
        image: str,
        env_vars: Optional[Dict[str, str]] = None,
        min_replicas: int = 1,
        max_replicas: int = 10,
        cpu: float = 1.0,
        memory: str = "2Gi",
    ) -> Dict:
        """Deploy or update the Container App."""
        try:
            from azure.mgmt.appcontainers.models import (
                ContainerApp,
                Configuration,
                Ingress,
                Template,
                Container,
                Scale,
                EnvironmentVar,
                RegistryCredentials,
            )

            containers = [
                Container(
                    name="llmops-api",
                    image=image,
                    resources={"cpu": cpu, "memory": memory},
                    env=[
                        EnvironmentVar(name=k, value=v)
                        for k, v in (env_vars or {}).items()
                    ],
                )
            ]
            app = ContainerApp(
                location=self.config.location,
                managed_environment_id=self._get_env_id(),
                configuration=Configuration(
                    ingress=Ingress(external=True, target_port=8000),
                    registries=[
                        RegistryCredentials(
                            server=f"{self.config.acr_name}.azurecr.io",
                            username=os.getenv("ACR_USERNAME", ""),
                            password_secret_ref="acr-password",
                        )
                    ],
                ),
                template=Template(
                    containers=containers,
                    scale=Scale(min_replicas=min_replicas, max_replicas=max_replicas),
                ),
            )
            result = self._client.container_apps.begin_create_or_update(
                self.config.resource_group, self.config.container_app_name, app
            ).result()
            fqdn = getattr(
                getattr(getattr(result, "properties", None), "configuration", None),
                "ingress", type("", (), {"fqdn": "pending"})()
            ).fqdn
            url = f"https://{fqdn}"
            logger.info("Container App deployed: %s → %s", self.config.container_app_name, url)
            return {"app_name": self.config.container_app_name, "url": url}
        except Exception as e:
            logger.error("Container App deployment failed: %s", e)
            return {"app_name": self.config.container_app_name, "error": str(e)}


# ---------------------------------------------------------------------------
# Azure OpenAI Backend
# ---------------------------------------------------------------------------

class AzureOpenAIBackend:
    """
    Azure OpenAI inference backend — drop-in replacement for the standard OpenAI client.

    Integrates with the existing LangGraph synthesizer as an alternative to GPT-4o-mini direct.
    Accepts a messages list directly (matching the chat completions API shape).
    """

    def __init__(self, config: Optional[AzureConfig] = None):
        cfg = config or AzureConfig()
        self._client = None
        self.deployment = cfg.openai_deployment
        self._endpoint = cfg.openai_endpoint
        self._api_key = cfg.openai_api_key
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version="2024-08-01-preview",
            )
            logger.info("Azure OpenAI backend: %s | %s", self._endpoint, self.deployment)
        except ImportError:
            logger.warning("openai package not installed — Azure OpenAI backend unavailable")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> Union[str, Iterator]:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            stream:   If True, returns a streaming iterator of chunk strings.

        Returns:
            Response content string, or a streaming iterator when stream=True.
        """
        if self._client is None:
            raise RuntimeError("Azure OpenAI client not initialised")
        response = self._client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            return response  # caller iterates chunks
        return response.choices[0].message.content

    def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
    ) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI."""
        if self._client is None:
            raise RuntimeError("Azure OpenAI client not initialised")
        response = self._client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Azure Monitor
# ---------------------------------------------------------------------------

class AzureMonitor:
    """
    Emit custom metrics via the Azure Monitor custom metrics REST API.
    Mirrors infra/aws_observability.py (CloudWatch PutMetricData).
    """

    def __init__(self, config: Optional[AzureConfig] = None):
        cfg = config or AzureConfig()
        self._subscription_id = cfg.subscription_id
        self._resource_group = cfg.resource_group
        self._location = cfg.location
        self._credential = None
        self._init_credential()

    def _init_credential(self) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            self._credential = DefaultAzureCredential()
        except ImportError:
            logger.warning("azure-identity not installed — Azure Monitor metrics disabled")

    def emit_metric(
        self,
        metric_name: str,
        value: float,
        namespace: str = "LLMOps",
        dimensions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a custom metric to Azure Monitor."""
        if self._credential is None:
            logger.debug("Metric (no-op): %s=%.4f", metric_name, value)
            return
        try:
            import requests
            body = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": {
                    "baseData": {
                        "metric": metric_name,
                        "namespace": namespace,
                        "dimNames": list((dimensions or {}).keys()),
                        "series": [{
                            "dimValues": list((dimensions or {}).values()),
                            "min": value,
                            "max": value,
                            "sum": value,
                            "count": 1,
                        }],
                    }
                },
            }
            token = self._credential.get_token("https://monitor.azure.com/").token
            resource_id = (
                f"/subscriptions/{self._subscription_id}"
                f"/resourceGroups/{self._resource_group}"
            )
            url = f"https://{self._location}.monitoring.azure.com{resource_id}/metrics"
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                data=json.dumps(body),
                timeout=10,
            )
            resp.raise_for_status()
            logger.debug("Emitted metric %s=%.4f", metric_name, value)
        except Exception as e:
            logger.error("Azure Monitor metric emission failed: %s", e)

    def track_latency(self, operation: str, latency_ms: float) -> None:
        self.emit_metric(f"{operation}_latency_ms", latency_ms, dimensions={"operation": operation})

    def track_request(self, endpoint: str, status_code: int) -> None:
        self.emit_metric("api_requests", 1.0, dimensions={"endpoint": endpoint, "status": str(status_code)})

    def track_ragas_score(self, metric: str, score: float) -> None:
        self.emit_metric(f"ragas_{metric}", score, namespace="LLMOps/RAGAS", dimensions={"metric": metric})

    def track_model_inference(self, model: str, tokens: int, latency_ms: float) -> None:
        self.emit_metric("inference_tokens", float(tokens), dimensions={"model": model})
        self.emit_metric("inference_latency_ms", latency_ms, dimensions={"model": model})


# ---------------------------------------------------------------------------
# Mock clients for environments without Azure SDK
# ---------------------------------------------------------------------------

class _MockMLClient:
    class _Models:
        def list(self): return []
        def create_or_update(self, m):
            m.version = "1"
            return m
    class _Endpoints:
        def begin_create_or_update(self, e):
            class _P:
                def wait(self): pass
                def result(self): return e
            return _P()
        def get(self, name): return type("E", (), {"traffic": {}})()
    class _Deployments:
        def begin_create_or_update(self, d):
            class _P:
                def wait(self): pass
                def result(self): return d
            return _P()
    models = _Models()
    online_endpoints = _Endpoints()
    online_deployments = _Deployments()


class _MockACAClient:
    class _Apps:
        def begin_create_or_update(self, rg, name, app):
            class _P:
                def result(self): return type("R", (), {"properties": None})()
            return _P()
    class _Envs:
        def get(self, rg, name): return type("E", (), {"id": f"/mock/env/{name}"})()
    container_apps = _Apps()
    managed_environments = _Envs()


# ---------------------------------------------------------------------------
# Terraform template
# ---------------------------------------------------------------------------

AZURE_TERRAFORM = '''
# Azure infrastructure for LLMOps Research Assistant
# Mirrors infra/terraform/main.tf (AWS) — adds Azure path
#
# Deploy:
#   cd infra/azure
#   terraform init && terraform apply

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "llmops" {
  name     = var.resource_group
  location = var.location
}

resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.llmops.name
  location            = azurerm_resource_group.llmops.location
  sku                 = "Standard"
  admin_enabled       = true
}

resource "azurerm_machine_learning_workspace" "aml" {
  name                    = var.workspace_name
  location                = azurerm_resource_group.llmops.location
  resource_group_name     = azurerm_resource_group.llmops.name
  application_insights_id = azurerm_application_insights.llmops.id
  key_vault_id            = azurerm_key_vault.llmops.id
  storage_account_id      = azurerm_storage_account.llmops.id
  identity { type = "SystemAssigned" }
}

resource "azurerm_container_app_environment" "llmops" {
  name                       = var.container_app_env
  location                   = azurerm_resource_group.llmops.location
  resource_group_name        = azurerm_resource_group.llmops.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.llmops.id
}

resource "azurerm_container_app" "api" {
  name                         = "llmops-api"
  container_app_environment_id = azurerm_container_app_environment.llmops.id
  resource_group_name          = azurerm_resource_group.llmops.name
  revision_mode                = "Single"

  template {
    container {
      name   = "llmops-api"
      image  = "${azurerm_container_registry.acr.login_server}/llmops-api:latest"
      cpu    = 1.0
      memory = "2Gi"
    }
    min_replicas = 1
    max_replicas = 10
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

resource "azurerm_cognitive_account" "openai" {
  name                = "llmops-openai"
  location            = azurerm_resource_group.llmops.location
  resource_group_name = azurerm_resource_group.llmops.name
  kind                = "OpenAI"
  sku_name            = "S0"
}

resource "azurerm_cognitive_deployment" "gpt4o_mini" {
  name                 = "gpt-4o-mini"
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = "gpt-4o-mini"
    version = "2024-07-18"
  }
  scale { type = "Standard" }
}

resource "azurerm_key_vault" "llmops" {
  name                = var.key_vault_name
  location            = azurerm_resource_group.llmops.location
  resource_group_name = azurerm_resource_group.llmops.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
}

resource "azurerm_application_insights" "llmops" {
  name                = "llmops-appinsights"
  location            = azurerm_resource_group.llmops.location
  resource_group_name = azurerm_resource_group.llmops.name
  application_type    = "web"
}

resource "azurerm_log_analytics_workspace" "llmops" {
  name                = "llmops-logs"
  location            = azurerm_resource_group.llmops.location
  resource_group_name = azurerm_resource_group.llmops.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_storage_account" "llmops" {
  name                     = "llmopssa"
  resource_group_name      = azurerm_resource_group.llmops.name
  location                 = azurerm_resource_group.llmops.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

variable "resource_group"     { default = "llmops-rg" }
variable "location"           { default = "eastus" }
variable "acr_name"           { default = "llmopsacr" }
variable "workspace_name"     { default = "llmops-aml" }
variable "container_app_env"  { default = "llmops-env" }
variable "key_vault_name"     { default = "llmops-kv" }

data "azurerm_client_config" "current" {}

output "api_url"          { value = azurerm_container_app.api.latest_revision_fqdn }
output "openai_endpoint"  { value = azurerm_cognitive_account.openai.endpoint }
output "acr_login_server" { value = azurerm_container_registry.acr.login_server }
'''


def write_terraform(output_dir: str = "./infra/azure") -> None:
    """Write the Terraform template to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "main.tf")
    with open(path, "w") as f:
        f.write(AZURE_TERRAFORM.lstrip())
    logger.info("Azure Terraform written to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = AzureConfig()

    logger.info("=== Azure ML Manager demo ===")
    ml = AzureMLManager(config)
    version = ml.register_model("./models/rag_model", "llmops-rag", description="RAG pipeline model")
    logger.info("Registered version: %s", version)

    logger.info("=== Azure Container Apps demo ===")
    aca = AzureContainerAppsDeployer(config)
    result = aca.deploy(
        f"{config.acr_name}.azurecr.io/llmops-api:latest",
        env_vars={"OPENAI_API_KEY": "placeholder"},
    )
    logger.info("Deploy result: %s", result)

    logger.info("=== Azure OpenAI Backend demo ===")
    aoai = AzureOpenAIBackend(config)
    logger.info("Azure OpenAI configured: endpoint=%s deployment=%s",
                config.openai_endpoint or "(not set)", config.openai_deployment)

    logger.info("=== Azure Monitor demo ===")
    monitor = AzureMonitor(config)
    monitor.track_latency("retrieval", 42.5)
    monitor.track_ragas_score("faithfulness", 0.847)
    logger.info("Metrics emitted (no-op if SDK not installed or credentials absent)")

    logger.info("=== Writing Terraform template ===")
    write_terraform()

    logger.info("Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_OPENAI_ENDPOINT to use live Azure.")


if __name__ == "__main__":
    main()
