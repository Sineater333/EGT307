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
        echo -e "Creating .env from template... please edit it with your Atlas URL."
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

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Please create one with ATLAS_URL."
    exit 1
fi

echo -e "${CYAN}--- Syncing Kubernetes Secrets ---${NC}"

# Check if secret exists and delete it to ensure it updates with the latest .env values
kubectl delete secret mongodb-atlas-secret --ignore-not-found

# Create the secret directly from the file
kubectl create secret generic mongodb-atlas-secret --from-env-file=.env

echo -e "${GREEN}Secrets synced from .env file successfully.${NC}"

echo -e "${CYAN}--- Applying Manifests ---${NC}"
kubectl apply -k ./k8s-manifests/

echo -e "${YELLOW}--- Waiting for Pods to be Ready ---${NC}"
sleep 15
kubectl wait --for=condition=Ready pods --all --timeout=600s || {
    echo -e "${YELLOW}Wait timed out, but pods may still be starting. Checking status...${NC}"
    kubectl get pods
}

echo -e "${CYAN}--- Launching Kubernetes Admin Dashboard ---${NC}"
# Open minikube dashboard in background (non-blocking)
minikube dashboard &
DASHBOARD_PID=$!
sleep 3

echo -e "${CYAN}--- Enabling Ingress Controller LoadBalancer & Starting Tunnel ---${NC}"
# Start minikube tunnel in background (required for LoadBalancer external IP on Minikube)
nohup minikube tunnel > /tmp/minikube-tunnel.log 2>&1 &
TUNNEL_PID=$!
sleep 2

echo -e "${YELLOW}--- Waiting for LoadBalancer external IP (polling up to 60 seconds) ---${NC}"
EXT_IP=""
for i in {1..60}; do
  EXT_IP=$(kubectl get svc ingress-nginx-lb -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  if [ -n "$EXT_IP" ]; then
    echo -e "${GREEN}✓ LoadBalancer assigned external IP: $EXT_IP${NC}"
    break
  fi
  echo -n "."
  sleep 1
done

if [ -z "$EXT_IP" ]; then
  echo -e "${YELLOW}⚠  External IP not yet assigned (may still be pending).${NC}"
  echo -e "${YELLOW}   This is OK if minikube tunnel is running. Check 'kubectl get svc -n ingress-nginx' for status.${NC}"
  EXT_IP="<PENDING>"
fi

echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e ""
echo -e "${CYAN}=== NEXT STEPS ===${NC}"
echo -e "${YELLOW}Open in browser:${NC}"
echo -e "   ${GREEN}http://maintenance.local/${NC} (Dashboard)"
echo -e "   ${GREEN}http://maintenance.local/api/docs${NC} (API Gateway)"
echo -e ""
echo -e "${YELLOW}Dashboard and tunnel are running in the background.${NC}"
echo -e "${YELLOW}To check status: ${NC}kubectl get svc -n ingress-nginx"