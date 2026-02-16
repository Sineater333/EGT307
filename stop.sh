#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set +e 

# Colors
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}--- Deleting Kubernetes Resources ---${NC}"
# Delete everything in the kustomization
kubectl delete -k ./k8s-manifests/

echo -e "${YELLOW}--- Removing Cloud Secrets ---${NC}"
kubectl delete secret mongodb-atlas-secret

echo -e "${CYAN}--- Shutting Down Minikube Cluster ---${NC}"
echo "Freezing minikube to save RAM..."
minikube pause

echo -e "${CYAN}--- Cleaning up Local Tunnels ---${NC}"
# Use standard taskkill to clean up minikube processes on Windows
taskkill //F //IM minikube.exe //T 2>/dev/null || echo "No lingering tunnels found."

echo -e "${GREEN}System Stopped Successfully.${NC}"