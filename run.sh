#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for better readability
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

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
minikube start --driver=docker --cpus 4 --memory 4096 --wait=all

echo -e "${YELLOW}--- Linking Context ---${NC}"
minikube update-context
kubectl config use-context minikube

echo -e "${CYAN}--- Enabling Metrics Server ---${NC}"
minikube addons enable metrics-server

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Please create one with ATLAS_URL."
    exit 1
fi

echo -e "${CYAN}--- Injecting MongoDB Secret ---${NC}"
kubectl create secret generic mongodb-atlas-secret \
  --from-literal=atlas-url="$ATLAS_URL" \
  --dry-run=client -o yaml | kubectl apply -f -

echo -e "${CYAN}--- Applying Manifests ---${NC}"
kubectl apply -k ./k8s-manifests/

echo -e "${YELLOW}--- Waiting for Pods to be Ready ---${NC}"
sleep 10
kubectl wait --for=condition=Ready pods --all --timeout=120s 

echo -e "${CYAN}Launching Admin Dashboard...${NC}"
# Use & to run in the background so the script can continue to the next command
minikube dashboard &

echo -e "${GREEN}Launching Application UI...${NC}"
sleep 5
# This will open the browser and keep the connection open
minikube service dashboard-service