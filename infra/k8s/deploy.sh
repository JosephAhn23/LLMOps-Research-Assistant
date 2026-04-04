#!/bin/bash
# Full cluster deployment script
set -e

echo "Creating namespace..."
kubectl apply -f infra/k8s/namespace.yaml

echo "Creating configmap..."
kubectl apply -f infra/k8s/configmap.yaml

echo "Creating/updating secret manifest..."
kubectl apply -f infra/k8s/secret.yaml

echo "Deploying Redis..."
kubectl apply -f infra/k8s/redis.yaml

echo "Deploying API (3 replicas + HPA)..."
kubectl apply -f infra/k8s/api-deployment.yaml

echo "Deploying workers..."
kubectl apply -f infra/k8s/worker-deployment.yaml

echo "Deploying GPU fine-tune job..."
kubectl apply -f infra/k8s/gpu-deployment.yaml

echo "Deploying HPA..."
kubectl apply -f infra/k8s/hpa.yaml

echo "Waiting for rollout..."
kubectl rollout status deployment/llmops-api -n llmops
kubectl rollout status deployment/llmops-worker -n llmops

echo "All services deployed."
kubectl get all -n llmops
