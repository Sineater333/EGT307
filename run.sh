#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for better readability
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

NAMESPACE="egt307-app"

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

# --- NEW: CREATE NAMESPACE ---
echo -e "${CYAN}--- Ensuring Namespace '${NAMESPACE}' exists ---${NC}"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo -e "${CYAN}--- Enabling Metrics Server ---${NC}"
minikube addons enable metrics-server

echo -e "${CYAN}--- Enabling Ingress Addon ---${NC}"
minikube addons enable ingress || true

echo -e "${CYAN}--- Cleaning up old Ingress State ---${NC}"
# Note: Added -n $NAMESPACE to ensure we clean the right area
kubectl delete ingress api-ingress dashboard-ingress docs-ingress -n $NAMESPACE --ignore-not-found --now
kubectl delete validatingwebhookconfiguration ingress-nginx-admission --ignore-not-found

echo -e "${CYAN}--- Syncing Kubernetes Secrets ---${NC}"

# --- UPDATED: TARGET SPECIFIC NAMESPACE ---
kubectl delete secret mongodb-atlas-secret -n $NAMESPACE --ignore-not-found
kubectl create secret generic mongodb-atlas-secret --from-env-file=.env -n $NAMESPACE

echo -e "${GREEN}Secrets synced to namespace '${NAMESPACE}' successfully.${NC}"

echo -e "${CYAN}--- Waiting for Ingress Controller to be Ready ---${NC}"
# Ensure ingress controller is running before applying ingress resources (prevents webhook race)
kubectl wait -n ingress-nginx \
  --for=condition=ready pod \
  -l app.kubernetes.io/component=controller \
  --timeout=180s || {
    echo -e "${YELLOW}⚠ Ingress controller not ready yet. Continuing anyway...${NC}"
  }

echo -e "${CYAN}--- Applying Manifests (Kustomize) ---${NC}"
# Kustomize will use the 'namespace: egt307-app'
kubectl apply -k ./k8s-manifests/ || {
  echo -e "${YELLOW}⚠ Apply failed. Retrying after 10s...${NC}"
  sleep 10
  kubectl apply -k ./k8s-manifests/
}
# delete the ghost service Kustomize just created in the app namespace
kubectl delete svc ingress-nginx-lb -n $NAMESPACE --ignore-not-found

# Force the LoadBalancer Service into the correct namespace
kubectl apply -f ./k8s-manifests/ingress-nginx-lb.yaml -n ingress-nginx
echo -e "${GREEN}✓ LoadBalancer shifted to ingress-nginx namespace.${NC}"

# echo -e "2. If 'minikube tunnel' is already running, ${YELLOW}STOP IT (Ctrl+C)${NC} and ${GREEN}RESTART IT${NC} now."
# echo -e "3. Waiting 15 seconds for NGINX to load rules..."
# sleep 15

echo -e "${YELLOW}--- Checking Workloads in ${NAMESPACE} ---${NC}"
kubectl get pods -n $NAMESPACE
kubectl get svc -n $NAMESPACE
kubectl get ingress -n $NAMESPACE || true

echo -e "${YELLOW}--- Setting Default Namespace to egt307-app ---${NC}"
kubectl config set-context --current --namespace=egt307-app

echo -e "${GREEN}✓ Context switched. 'kubectl get pods' will now show egt307-app by default.${NC}"

echo -e "${CYAN}--- Launching Kubernetes Dashboard (optional) ---${NC}"
# Dashboard is optional; on Windows it may open a browser automatically
taskkill //F //IM "kubectl.exe" //T 2>/dev/null || true

nohup minikube tunnel > /dev/null 2>&1 &

echo -e "${GREEN}✓ Tunnel is running in the background.${NC}"

# 2. Start the dashboard in a completely detached way
# We use 'nohup' and '&' to ensure it doesn't hold the terminal hostage
nohup minikube dashboard --url > dashboard_url.txt 2>&1 &

# 3. Wait for the proxy to generate the URL in the text file
echo "Waiting for dashboard proxy to initialize..."
sleep 5

# 4. Extract the URL and open it with the namespace fragment
DASH_URL=$(grep -o 'http://127.0.0.1:[0-9]*' dashboard_url.txt | head -n 1 || echo "http://127.0.0.1:8001")
FINAL_URL="${DASH_URL}/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/#/workloads?namespace=${NAMESPACE}"

echo -e "${GREEN}Opening Dashboard: ${FINAL_URL}${NC}"
powershell.exe -Command "Start-Process '$FINAL_URL'"

# Clean up the temp file
rm dashboard_url.txt

echo -e "${CYAN}--- LoadBalancer + Tunnel Notes (Windows) ---${NC}"
echo -e "${YELLOW}Minikube LoadBalancer requires 'minikube tunnel'.${NC}"
echo -e "${YELLOW}On Windows, start it manually in an Administrator PowerShell and keep it open:${NC}"
echo -e "   ${GREEN}minikube tunnel${NC}"
echo -e ""

echo -e "${YELLOW}--- Checking LoadBalancer Status ---${NC}"
kubectl get svc -n ingress-nginx ingress-nginx-lb || true

echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e ""
echo -e "${CYAN}=== ACCESS URLS ===${NC}"
echo -e "   ${GREEN}http://maintenance.local/${NC} (Dashboard UI)"
echo -e "   ${GREEN}http://maintenance.local/docs${NC} (API Swagger UI)"
echo -e "   ${GREEN}http://maintenance.local/openapi.json${NC} (OpenAPI JSON)"
echo -e "   ${GREEN}http://maintenance.local/api/health${NC} (API via /api prefix, if configured)"
echo -e ""
echo -e "${CYAN}=== QUICK DEBUG COMMANDS ===${NC}"
echo -e "   kubectl get pods -n $NAMESPACE"
echo -e "   kubectl logs deploy/gateway-deployment -n $NAMESPACE --tail=80"