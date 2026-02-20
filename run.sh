#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for better readability
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== EGT307 Kubernetes Deployment Script ===${NC}"
echo -e ""

echo -e "${CYAN}--- Loading Environment Variables ---${NC}"

# Check if .env exists, if not, check for .env.example
if [ ! -f .env ]; then
    echo -e "${YELLOW}Missing .env file!${NC}"
    if [ -f .env.example ]; then
        echo -e "Creating .env from template... please edit it with your MongoDB URL."
        cp .env.example .env
    fi
    exit 1
fi

# Load .env without exporting comments
set -a
source .env
set +a

echo -e "${GREEN}Environment loaded successfully.${NC}"

echo -e "${CYAN}--- Starting Minikube ---${NC}"
# Check if minikube is paused, if so, unpause. If not running, start it.
if minikube status | grep -q "Paused"; then
    echo "Resuming minikube..."
    minikube unpause
else
    echo "Starting minikube fresh..."
    minikube start --driver=docker --memory=2200 --cpus=2 --wait=false
fi

echo -e "${YELLOW}--- Linking Context ---${NC}"
minikube update-context
kubectl config use-context minikube

echo -e "${CYAN}--- Enabling Metrics Server ---${NC}"
minikube addons enable metrics-server

echo -e "${CYAN}--- Enabling Ingress Addon ---${NC}"
minikube addons enable ingress || true

echo -e "${CYAN}--- Syncing Kubernetes Secrets ---${NC}"

# Check if secret exists and delete it to ensure it updates with the latest .env values
kubectl delete secret mongodb-atlas-secret --ignore-not-found

# Create the secret directly from the file
kubectl create secret generic mongodb-atlas-secret --from-env-file=.env

echo -e "${GREEN}Secrets synced from .env file successfully.${NC}"

echo -e "${CYAN}--- Waiting for Ingress Controller to be Ready ---${NC}"
# Ensure ingress controller is running before applying ingress resources (prevents webhook race)
kubectl wait -n ingress-nginx \
  --for=condition=ready pod \
  -l app.kubernetes.io/component=controller \
  --timeout=180s || {
    echo -e "${YELLOW}⚠ Ingress controller not ready yet. Continuing anyway, but ingress apply may fail.${NC}"
  }

echo -e "${CYAN}--- Applying Manifests (Kustomize) ---${NC}"
kubectl apply -k ./k8s-manifests/ || {
  echo -e "${YELLOW}⚠ Apply failed (often due to ingress webhook not ready). Retrying after 10s...${NC}"
  sleep 10
  kubectl apply -k ./k8s-manifests/
}

echo -e "${YELLOW}--- Checking Workloads ---${NC}"
kubectl get pods
kubectl get svc
kubectl get ingress || true

echo -e "${CYAN}--- Launching Kubernetes Dashboard (optional) ---${NC}"
# Dashboard is optional; on Windows it may open a browser automatically
minikube dashboard &>/dev/null &
sleep 2

echo -e "${CYAN}--- LoadBalancer + Tunnel Notes (Windows) ---${NC}"
echo -e "${YELLOW}Minikube LoadBalancer requires 'minikube tunnel'.${NC}"
echo -e "${YELLOW}On Windows, start it manually in an Administrator PowerShell and keep it open:${NC}"
echo -e "   ${GREEN}minikube tunnel${NC}"
echo -e ""

echo -e "${YELLOW}--- Checking LoadBalancer Status (may be <pending> until tunnel runs) ---${NC}"
kubectl get svc -n ingress-nginx ingress-nginx-lb || true
echo -e "${YELLOW}If EXTERNAL-IP is <pending>, run 'minikube tunnel' in an Admin PowerShell.${NC}"

echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e ""
echo -e "${CYAN}=== ACCESS URLS ===${NC}"
echo -e "   ${GREEN}http://maintenance.local/${NC} (Dashboard UI)"
echo -e "   ${GREEN}http://maintenance.local/docs${NC} (API Swagger UI)"
echo -e "   ${GREEN}http://maintenance.local/openapi.json${NC} (OpenAPI JSON)"
echo -e "   ${GREEN}http://maintenance.local/api/health${NC} (API via /api prefix, if configured)"
echo -e ""
echo -e "${CYAN}=== QUICK DEBUG COMMANDS ===${NC}"
echo -e "   kubectl get pods"
echo -e "   kubectl get ingress"
echo -e "   kubectl describe ingress"
echo -e "   kubectl logs deploy/api-gateway-deployment --tail=80"