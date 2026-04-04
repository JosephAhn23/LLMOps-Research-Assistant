# LLMOps Research Assistant — Azure Infrastructure (Terraform)
# Mirrors infra/terraform/main.tf (AWS) for full Azure parity.
#
# Resources:
#   - Resource Group
#   - Azure Container Registry (ACR)
#   - Azure ML Workspace
#   - Container Apps Environment + API/Worker apps
#   - Azure OpenAI (Cognitive Services)
#   - Key Vault
#   - Application Insights + Log Analytics
#
# Deploy:
#   cd infra/azure
#   terraform init
#   terraform apply -var="subscription_id=<your-sub-id>"

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
  subscription_id = var.subscription_id
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "llmops-rg"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  default     = "prod"
}

variable "api_image" {
  description = "Container image for the API service"
  type        = string
  default     = "llmopsacr.azurecr.io/llmops-api:latest"
}

variable "worker_image" {
  description = "Container image for the Celery worker"
  type        = string
  default     = "llmopsacr.azurecr.io/llmops-worker:latest"
}

variable "openai_sku" {
  description = "Azure OpenAI SKU tier"
  type        = string
  default     = "S0"
}

# ---------------------------------------------------------------------------
# Random suffix to ensure unique resource names
# ---------------------------------------------------------------------------

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

locals {
  suffix   = random_string.suffix.result
  acr_name = "llmopsacr${local.suffix}"
  kv_name  = "llmops-kv-${local.suffix}"
  tags = {
    project     = "llmops-research-assistant"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Resource Group
# ---------------------------------------------------------------------------

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.tags
}

# ---------------------------------------------------------------------------
# Log Analytics + Application Insights
# ---------------------------------------------------------------------------

resource "azurerm_log_analytics_workspace" "main" {
  name                = "llmops-logs-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "llmops-insights-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.tags
}

# ---------------------------------------------------------------------------
# Azure Container Registry
# ---------------------------------------------------------------------------

resource "azurerm_container_registry" "main" {
  name                = local.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  admin_enabled       = false
  tags                = local.tags
}

# ---------------------------------------------------------------------------
# Azure ML Workspace
# ---------------------------------------------------------------------------

resource "azurerm_machine_learning_workspace" "main" {
  name                    = "llmops-aml-${local.suffix}"
  location                = azurerm_resource_group.main.location
  resource_group_name     = azurerm_resource_group.main.name
  application_insights_id = azurerm_application_insights.main.id
  container_registry_id   = azurerm_container_registry.main.id
  storage_account_id      = azurerm_storage_account.main.id
  key_vault_id            = azurerm_key_vault.main.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# ---------------------------------------------------------------------------
# Storage Account (required by Azure ML)
# ---------------------------------------------------------------------------

resource "azurerm_storage_account" "main" {
  name                     = "llmopssa${local.suffix}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = local.tags
}

# ---------------------------------------------------------------------------
# Key Vault
# ---------------------------------------------------------------------------

data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                        = local.kv_name
  location                    = azurerm_resource_group.main.location
  resource_group_name         = azurerm_resource_group.main.name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
  sku_name                    = "standard"
  tags                        = local.tags

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    secret_permissions = ["Get", "List", "Set", "Delete", "Purge"]
  }
}

# ---------------------------------------------------------------------------
# Azure OpenAI (Cognitive Services)
# ---------------------------------------------------------------------------

resource "azurerm_cognitive_account" "openai" {
  name                = "llmops-aoai-${local.suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = var.openai_sku
  tags                = local.tags

  custom_subdomain_name = "llmops-aoai-${local.suffix}"
}

resource "azurerm_cognitive_deployment" "gpt4o" {
  name                 = "gpt-4o"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "gpt-4o"
    version = "2024-05-13"
  }

  scale {
    type     = "Standard"
    capacity = 10
  }
}

resource "azurerm_cognitive_deployment" "embeddings" {
  name                 = "text-embedding-ada-002"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "text-embedding-ada-002"
    version = "2"
  }

  scale {
    type     = "Standard"
    capacity = 10
  }
}

# ---------------------------------------------------------------------------
# Container Apps Environment
# ---------------------------------------------------------------------------

resource "azurerm_container_app_environment" "main" {
  name                       = "llmops-env-${local.suffix}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

# ---------------------------------------------------------------------------
# API Container App
# ---------------------------------------------------------------------------

resource "azurerm_container_app" "api" {
  name                         = "llmops-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type = "SystemAssigned"
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = "System"
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "http"

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  template {
    container {
      name   = "api"
      image  = var.api_image
      cpu    = 1.0
      memory = "2Gi"

      env {
        name        = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        secret_name = "appinsights-conn"
      }

      liveness_probe {
        path      = "/health"
        port      = 8000
        transport = "HTTP"
      }
    }

    min_replicas = 1
    max_replicas = 10

    http_scale_rule {
      name                = "http-scaling"
      concurrent_requests = "50"
    }
  }

  secret {
    name  = "appinsights-conn"
    value = azurerm_application_insights.main.connection_string
  }
}

# ---------------------------------------------------------------------------
# Worker Container App (scale-to-zero)
# ---------------------------------------------------------------------------

resource "azurerm_container_app" "worker" {
  name                         = "llmops-worker"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type = "SystemAssigned"
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = "System"
  }

  template {
    container {
      name    = "worker"
      image   = var.worker_image
      cpu     = 2.0
      memory  = "4Gi"
      command = ["celery", "-A", "ingestion.batch_processor", "worker", "--loglevel=info"]
    }

    min_replicas = 0
    max_replicas = 10
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "api_url" {
  description = "Public URL of the API Container App"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "acr_login_server" {
  description = "ACR login server for pushing images"
  value       = azurerm_container_registry.main.login_server
}

output "azure_openai_endpoint" {
  description = "Azure OpenAI endpoint URL"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "aml_workspace_id" {
  description = "Azure ML workspace resource ID"
  value       = azurerm_machine_learning_workspace.main.id
}

output "appinsights_connection_string" {
  description = "Application Insights connection string"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}
