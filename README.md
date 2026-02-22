
# EGT307 — README

## Project Overview and Objectives

### Domain: Machine Learning & Automation
This project addresses the critical issue of unexpected machine downtime in manufacturing which will be costly. By leveraging real-time sensor data and a Duo-stage machine learning pipeline, our system predicts failures early and identifies specific failure types (e.g., Tool Wear(`TWF`), Heat Dissipation(`HDF`)).

### Objectives 
* **Engineering Goal**: Change from reactive to proactive maintenance using AI-driven insights to prevent any downtime from happening.

* **System Goal**: Deploy a scalable, microservices-based independent services architecture orchestrated by Kubernetes.

* **Customer Value**: Provide factory managers with a "customer-centric" dashboard for real-time data monitoring and historical data analysis.

### Solution
To solve the identified problem, the system carries out through three key layers:

* **Data-Driven Intelligence**: We use the AI4I 2020 Predictive Maintenance Dataset to train the duo-stage model. The first stage detects if a failure has occurred or not, while the second stage will determine the specific failure causes (e.g., TWF, HDF, PWF, etc).

* **Modular Microservices**: The application will be split into four independent services; the API Gateway service, the inference Service, database Service, and dashboard service. This ensures that if one component fails, the rest of the system can remain operational (fault tolerance and modularity).

* **Scalable Orchestration**: The entire stack is containerized with docker and managed by Kubernetes. We implement Horizontal Pod Autoscaling (HPA) to automatically scale the inference service during periods of high sensor data throughput (Scalability).

## A. Group member name and email
- Ong Zhao Ji (joedoe369lol@gmail.com)
- Jonathan Ng Jia Xian (jiaxian2325@gmail.com)
- Tai Wei en (Taiweien06@gmail.com)

## B. Project structure and purpose

The project is divided into five independent microservices to ensure modularity and scalability:
```
└── 📁EGT307
    └── 📁api-gateway
        ├── Dockerfile
        ├── main.py
        ├── requirements.txt
    └── 📁dashboard-ui
        ├── dashboard.py
        ├── Dockerfile
        ├── requirements.txt
    └── 📁database-service
        ├── Dockerfile
        ├── main.py
        ├── requirements.txt
    └── 📁inference-service
        └── 📁logs
            ├── prediction_history.csv
        └── 📁models
            ├── binary_model.pkl
            ├── label_encoder.pkl
            ├── metrics.txt
            ├── model.pkl
            ├── type_model.pkl
        ├── Dockerfile
        ├── main.py
        ├── requirements.txt
        ├── train.py
    └── 📁k8s-manifests
        ├── dashboard-deployment.yaml
        ├── database-deployment.yaml
        ├── gateway-deployment.yaml
        ├── hpa.yaml
        ├── inference-deployment.yaml
        ├── ingress-dashboard.yaml
        ├── ingress-docs.yaml
        ├── ingress-nginx.yaml
        ├── ingress-api.yaml
        ├── kustomization.yaml
    ├── .dockerignore
    ├── .env.example
    ├── .gitignore
    ├── Docker-compose.yml
    ├── README.md
    ├── run.sh
    └── stop.sh
```

- **`api-gateway/`**: This service acts as the single entry point (Front Door) for all client requests.It exposes unified endpoints to the UI and internally orchestrates communication between services:
    - `main.py`: A FastAPI application that:
    	- Receives prediction requests from the dashboard
		- Forwards sensor data to the inference-service
		- Logs prediction results to the database-service
    	- Proxies historical data requests
	- `Dockerfile`: Containerizes the gateway for consistent deployment across environments.
	- `requirements.txt`: Minimal dependencies for high-performance routing (FastAPI, Uvicorn, Requests).


- **`dashboard-ui/`**: The user interface, designed for factory floor managers to monitor equipment health in real-time:
    - `dashboard.py`: Built with Streamlit. It features:
        * Input Forms: For manual entry of machinery data (Torque, RPM, Temp).
        * Sends prediction requests to the API Gateway
        * Retrieves historical logs via the API Gateway
        * Displays analytics and visualizations
	- `Dockerfile`: Sets up the Streamlit environment and exposes port 8501.
	- `requirements.txt`: Includes streamlit, pandas, and plotly for data visualization.

- **`database-service/`**: The persistence layer responsible for data storage:
	- `main.py`:
  		* Exposes /logs endpoint for storing prediction results
        * Exposes /history endpoint for retrieving recent logs
        * Connects to MongoDB Atlas
	- `Dockerfile`: Configures the service to run within the cluster network.
	- `requirements.txt`: Includes motor or pymongo for asynchronous database drivers.

- **`inference-service/`**: The AI engine responsible purely for machine failure prediction:
	- `train.py`: The training file for the model, implemented with SMOTE to handle class imbalance and exports the trained artifacts. Only Once.
	- `main.py`: The inference API. It executes the Two-Stage Logic:
        * Loads trained ML models
        * Executes Two-Stage logic:
        	* Stage 1: Binary classification (failure or not)
			* Stage 2: Failure type diagnosis (if it fail)
        * Returns prediction results to the API Gateway
	- `models/`:
        * binary_model.pkl: The model to determine if there is failure or not, optimized for high Recall.
        * type_model.pkl: The model to diagnose the specific type of failure.
        * label_enoder.pkl: Ensures categorical features are transformed consistently.
    - `logs/`: Contains `prediction_history.csv` used for local debugging.

- **`k8s-manifests/`**: Configuration files for deploying the system to Kubernetes cluster (Minikube).
	- `*-deployment.yaml`: Defines how each service run, including container image version, number of replicas, CPU/memory resource limits.
	- `hpa.yaml`: Horizontal Pod Autoscaler configuration that automatically scales the inference-service based on workload (e.g., CPU usage), up to 5 replicas during heavy computation.
    - `dashboard-deployment.yaml`: Deploys the Streamlit dashboard UI and exposes it internally via dashboard-service (ClusterIP). The dashboard communicates with the API Gateway through internal Kubernetes DNS.
    - `database-deployment.yaml`: Deploys the database-service (Python wrapper for MongoDB Atlas). Securely loads the MONGO_URL from an env file nd exposes it internally via database-service.
    - `gateway-deployment.yaml`: Deploys the FastAPI API Gateway, which acts as the central entry point for prediction, logging, and history APIs. Routes requests internally to inference-service and database-service.
    - `inference-deployment.yaml`: Deploys the ML inference microservice responsible for model predictions. Configured with multiple replicas for load handling and fault tolerance.
    - `ingress-dashboard.yaml`: Routes external traffic from maintenance.local/ to the dashboard UI service.
    - `ingress-docs.yaml`: Exposes FastAPI documentation endpoints (/docs, /openapi.json) through the API Gateway.
    - `ingress-api.yaml`: Routes external API requests from maintenance.local/api to the API Gateway service.
    - `ingress-nginx-lb.yaml`: Exposes the NGINX Ingress Controller as a LoadBalancer service in Minikube, allowing external access to the cluster via port 80/443.
	- `kustomization.yaml`:Groups all Kubernetes manifests into a single deployable unit using Kustomize (kubectl apply -k), simplifying cluster setup.


- **`.env.example`**: A template for environment variables (e.g., MONGO_URI). Actual .env files are ignored by git for security.

- **`Docker-compose.yml`**: Used for Local Development. It spins up the entire stack with a single command without needing Kubernetes.

- **`run.sh`**: The "Master setup" script using bash. It automates:
    * Infrastructure Provisioning: Starts a Minikube cluster with optimized resources (2 CPUs, 2.2GB RAM).
    * Kubernetes Configuration: Enables metrics-server (required for HPA) and ingress addon.
    * Dependency Management: Validates environment variables and synchronizes them into Kubernetes Secrets.
    * Kubernetes Orchestration: Applies the Kustomization manifests to deploy all microservices in order.
    * Ingress & Tunnel: Creates LoadBalancer service for the ingress controller and starts minikube tunnel in background.
    * Automated Access: Launches the Kubernetes Dashboard and makes the application accessible via http://maintenance.local/  

- **`stop.sh`**: A cleanup script to safely spin down the cluster and remove active deployments.

## Ci. How to Run: Docker Compose(Local Development)

- This is the fastest way to test the full system integration locally.

	**Prerequisite** : Ensure you are inside the unzipped project folder where docker-compose.yml is located in.

- Build and Run:
```
docker-compose up --build
```

- Access:
    * Dashboard: http://localhost:8501
    * API gateway docs: http://localhost:8080/docs

## Cii. How to Run: Kubernetes production (automated)

To deploy the entire stack automatically (Minikube, cluster setup, manifests, ingress, and tunnel) with a single command, use the provided automation script.

- **Prerequisites**:
    
1. Install required tools:
- Minikube
- Docker (or your preferred Minikube driver)
- kubectl

Verify in bash/powershell:
```bash
minikube version
kubectl version --client
docker version
```

2. Map IP address to local DNS : (So when user run "maintenance.local" it brings it to the ip address)

**Window:**

Run notepad in Administrator and open the file :

 `C:\Windows\System32\drivers\etc\hosts` 

Add the following line to your hosts file: 

        127.0.0.1   maintenance.local

Save the file.

3. Configure environment variables

Navigate to directory that contains `.env.example` :

Copy the example file 

    ```bash
    cp .env.example .env
    ```

Update your MongoDB Atlas URL in .env (example): (actual link in the presentation slide)

        ATLAS_URL=mongodb+srv://<USERNAME>:<PASSWORD>@<CLUSTER_ADDRESS>/maintenance_db?retryWrites=true&w=majority&appName=EGT307

- **Launch the Stack**:

Run the setup script on Git Bash (Windows) or bash (macOS/Linux) from the project root directory: 

```bash
./run.sh
```

- **What the script does**:
    
    1. Cluster Setup: Starts Minikube with 2 CPUs and 2.2GB RAM.

    2. Addons: Enables metrics-server (required for HPA) and ingress addon.

    3. Security: Creates Kubernetes secret from your `.env` file.

    4. Deployment: Applies all manifests via Kustomize (including ingress and LoadBalancer service).

    5. Tunnel: Starts `minikube tunnel` in the background to expose the LoadBalancer (may request admin/sudo privileges).

    6. Polling: Waits for the LoadBalancer to receive an external IP (up to 60 seconds).

- **If the App Is Not Reachable (Window)**:

    1. Start the tunnel manually in powershell and keep it open:
    ```powershell
    minikube tunnel
    ```

    2. Confirm the LoadBalancer is assigned an External IP:
    ```bash
    kubectl get svc -n ingress-nginx ingress-nginx-lb
    ```

- **After the Script Completes**:

    The tunnel (and any background processes started by the script) will remain running.

    Access the application:

    ***Dashboard (UI)***

    http://maintenance.local/

    ***API Gateway (Swagger UI)***

    http://maintenance.local/docs

    ***OpenAPI JSON***

    http://maintenance.local/openapi.json

    ***API endpoints via /api prefix***

    Example health check: http://maintenance.local/api/health


- **Shutdown & Cleanup**

    To safely stop all services and clean up resources:

    ```bash
    ./stop.sh
    ```

    This stops Minikube and deletes all deployments and secrets.    

## E. Dataset information and sources

This project uses the AI4I 2020 Predictive Maintenance Dataset for model training. It is a synthetic dataset that closely reflects on real world industrial maintenance data. It is designed to help build models to predict machine failure based on sensor readings.

### source

* Provider: Stephan Matzka, School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin.
* Repository: [Predictive Maintenance Dataset (AI4I 2020)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data)
* Citation: Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. Third International Conference on Artificial Intelligence for Industries (AI4I).

### Dataset Composition

The dataset consists of 10,000 data points (rows) and 14 features (columns). 

| Feature Name  | Description |
| ------------- |:-------------:|
| UDI      | unique identifier ranging from 1 to 10000     |
| Product ID      | Consists of a letter (L, M, or H for low, medium, and high quality variants) and a serial number.     |
| Type      | The quality variant of the product (Low, Medium, High)     |
| Air temperature [K]      | generated using a random walk process later normalized to a standard deviation of 2 K around 300 K     |
| Process temperature [K]      | generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.     |
| Rotational speed [rpm]      | calculated from a power of 2860 W, overlaid with a normally distributed noise     |
| Torque [Nm]      |  torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values     |
| Tool wear [min]      | The time (in minutes) the tool has been used in the process     |
| Machine failure      |  indicates whether the machine has failed in this particular datapoint for any of the following failure modes are true     |

### Failure Modes

The unique aspects of this dataset is that it identifies five independent failure modes:

* Tool Wear Failure (TWF): The tool is replaced or fails at a specific wear time.

* Heat Dissipation Failure (HDF): Driven by the difference between air and process temperature.

* Power Failure (PWF): Occurs when the power required for the process is insufficient.

* Overstrain Failure (OSF): Occurs if the product of torque and tool wear exceeds certain limits.

* Random Failures (RNF): Each process has a 0.1% chance of random failure.

### Model Architecture & Methodology

This project uses a duo stage predictive maintenance pipeline to handle class imbalance and provide specific diagnostic insights.

#### Stage 1: Binary Classification (Detection of Failure)

* Goal: Determine if a machine failure has occurred (Machine failure = 1)

* Model: Random Forest Classifier.

* Data Balancing: Since failures are rare in the raw data, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training set to prevent model being bias toward the "No Failure" class.

* Optimization: A custom threshold of 0.55 (instead of the default 0.5) was implemented to reduce "False Alarms," ensuring higher precision for maintenance alerts.

#### Stage 2: Specialist Classification

* Goal: If a failure is detected, identify the specific failure that causes it

* Scope: This model is trained exclusively only on rows where a failure occurred.

* Target Labels:
    * `TWF` (Tool Wear Failure)

    * `HDF` (Heat Dissipation Failure)

    * `PWF` (Power Failure)

    * `OSF` (Overstrain Failure)

* Mechanism: The model uses **idxmax()** to merge multiple binary failure flags into a single multi-class target for precise diagnosis.

### Data Preprocessing

To ensure model quality even though the dataset is clean, the following steps were taken:

* Feature dropping: Removed `UDI` and `Product ID` as they are unique identifiers with no predictive power.

* Noise removal: Dropped the `RNF` (Random Failure) column as it represents stochastic noise that does not correlate with sensor readings.

* Categorical encoding: The `Type` feature (L, M, H) was transformed using label encoding to make it compatible with the random forest algorithm.


## F. Issues and limitations

1. This project was deployed on minikube instead of a cloud-nmanaged kubernetes environment.

2. Ingress configuration and local DNS mapping :

To access services via maintenance.local, local host mapping was required.

**Issues faced:**

    - Had to manually edit Windows hosts file as administrator
    - Ingress controller required Minikube tunnel
    - Port forwarding sometimes conflicted with existing services

Ingress routing is simulated locally and depends on manual configuration.In a real production, a cloud load balancer and proper DNS management would replace this setup.

3. MangoDB Architecture Decision:

For Local MongoDB with PVC, the database state is bound to a single-node minikube cluster, data persistence is limited to host machine. To make the system support cross-machine state sharing a cloud database Mongodb Atlas was being used.

**Issue Faced:**

    - Atlas require environment handling (env.)

4. Ensuring loose coupling achitecture:

Multiple services were accessing the database directly, creating a tangled architecture that was difficult to debug and scale,we had to move the functions to api-gateway so communication is clean and single entry point for all data flow.
