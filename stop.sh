#!/bin/bash
set +e

YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${YELLOW}--- Deleting Kubernetes Resources (Kustomize) ---${NC}"
kubectl delete -k ./k8s-manifests/ --ignore-not-found

echo -e "${YELLOW}--- Removing Secrets ---${NC}"
kubectl delete secret mongodb-atlas-secret --ignore-not-found

echo -e "${CYAN}--- Stopping Minikube ---${NC}"
# Pick ONE behavior: stop (fully) or pause (keep state, lower RAM)
minikube stop

echo -e "${CYAN}--- Tunnel Notes (Windows) ---${NC}"
echo -e "${YELLOW}If you started 'minikube tunnel' manually in an Administrator PowerShell, close that PowerShell window to stop the tunnel.${NC}"

echo -e "${GREEN}✓ Stopped successfully.${NC}"