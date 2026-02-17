
# EGT307 — README

## Overview and objectives

This repository contains a modular, containerized microservices architecture designed for real-time machine failure prediction. The system utilizes a Two-Stage Machine Learning Pipeline to first identify if a failure is occurring and then diagnose the specific engineering cause (e.g., HDF, PWF, etc).

The goal of this project is to provide a "customer-centric" dashboard for factory managers to monitor equipment health, view historical failure trends stored in a Cloud Database (MongoDB Atlas), and ensure high availability through Kubernetes (Minikube) orchestration with auto-scaling capabilities.

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
        ├── ingress.yaml
        ├── kustomization.yaml
    ├── .dockerignore
    ├── .env.example
    ├── .gitignore
    ├── Docker-compose.yml
    ├── README.md
    ├── run.sh
    └── stop.sh
```

- **`api-gateway/`**: This service acts as the "Front door" for all client requests, providing a single endpoint for the UI while routing traffic to internal services:
    - `main.py`: A FastAPI application that validates incoming sensor data and routes requests to the `inference-service` or `database-service`.
	- `Dockerfile`: Containerizes the gateway for consistent deployment across environments.
	- `requirements.txt`: Minimal dependencies for high-performance routing (FastAPI, Uvicorn, Requests).

- **`dashboard-ui/`**: The user interface, designed for factory floor managers to monitor equipment health in real-time:
    - `dashboard.py`: Built with Streamlit. It features:
        * Input Forms: For manual entry of machinery data (Torque, RPM, Temp).
        * Visualization: Results showing failure probability, failure causes and historical data.
	- `Dockerfile`: Sets up the Streamlit environment and exposes port 8501.
	- `requirements.txt`: Includes streamlit, pandas, and plotly for data visualization.

- **`database-service/`**: The persistence layer, this manages all interactions with the Cloud Database to ensure data integrity and persistence:
	- `main.py`: A FastAPI wrapper for MongoDB Atlas. It handles logging every prediction and its associated sensor data.
	- `Dockerfile`: Configures the service to run within the cluster network.
	- `requirements.txt`: Includes motor or pymongo for asynchronous database drivers.

- **`inference-service/`**: The core containing the AI with the Two-Stage ML logic:
	- `train.py`: The training file for the model, implemented with SMOTE to handle class imbalance and exports the trained artifacts.
	- `main.py`: The inference API. It executes the Two-Stage Logic:
        * Stage 1: Binary classification (Is there a failure or not?).
        * Stage 2: If Failure is detected, triggers the second model to diagnose the specific type.
	- `models/`:
        * binary_model.pkl: The model to determine if there is failure or not, optimized for high Recall.
        * type_model.pkl: The model to diagnose the specific type of failure.
        * label_enoder.pkl: Ensures categorical features are transformed consistently.
    - `logs/`: Contains `prediction_history.csv` used for local debugging.

- **`k8s-manifests/`**: Configuration files for deploying the system to Kubernetes cluster (Minikube).
	- `*-deployment.yaml`: Defines the desired state, replicas, and container images for each microservice.
	- `hpa.yaml`: The Horizontal Pod Autoscaler(HPA) configuration, allowing the `inference-service` to scale up to 5 pods during heavy compute loads.
	- `kustomization.yaml`: Orchestrates the deployment of all manifests as a single logical unit.

- **`.env.example`**: A template for environment variables (e.g., MONGO_URI). Actual .env files are ignored by git for security.

- **`Docker-compose.yml`**: Used for Local Development. It spins up the entire stack with a single command without needing Kubernetes.

- **`run.sh`**: The "Master setup" script using bash. It automates:
    * Infrastructure Provisioning: Starts a Minikube cluster with optimized resources (4 CPUs, 4GB RAM).

    * Dependency Management: Validates environment variables and synchronizes them into Kubernetes Secrets.

    * Kubernetes Configuration: Enables the metrics-server (required for the HPA to function).

    * Kubernetes Orchestration: Applies the Kustomization manifests to deploy all microservices in order.

    * Automated Access: Triggers the Minikube Dashboard and tunnels the dashboard-service for immediate browser access.  

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

## Cii. How to Run: Kubernetes production (manual)

- If you prefer start up kubernetes manually, use the method below.

	Initial step: Start minikube:

		minikube start

    enable metrics for Horizontal Pod Autoscaler:

        minikube addons enable metrics-server

- Setup secrets 

    To allow Kubernetes to securely communicate with your cloud database:

    1. Create a `.env` file based on the template:
    
            cp .env.example .env
        
    2. Create the Kubernetes secret from your environment file:

            kubectl create secret generic mongodb-atlas-secret --from-env-file=.env
    
- Deploy Manifests

    1. Apply the configuration: Use the -k flag to deploy the entire stack in the correct order from kustomization.yaml.

            kubectl apply -f k8s-manifests/

    2. Check Pods and ensure all are running

            kubectl get pods -w
    
    3. Check HPA status (It may take 1-2 minutes to show '0%/50%')

            kubectl get hpa

- Access the Application

    Because Minikube runs in a virtualized environment, a tunnel is needed to access the UI:

    1. Open the Dashboard:

            minikube service dashboard-service

    2. (Optional) Monitor via K8s Dashboard: To see a visual representation of your cluster health:
    
            minikube dashboard

- Clean up 

    To stop the services and remove all resources created by the manifests:

        kubectl delete -f k8s-manifests/
        kubectl delete secret mongodb-atlas-secret
        minikube pause # use minikube stop if u want to completely shuts down the container running the Kubernetes cluster. 

## Ciii. How to Run: Kubernetes production (automated)

If you want to deploy the entire stack, including cluster provisioning and secrets management with a single command, use the provided automation script.

- **Prerequisite** : 
    
    1. Ensure Minikube and Docker (or your preferred driver) are installed.

    2. Copy .env.example to .env and update it with your mongo url.

- **Launch the Stack**:
Run the setup script on git bash from the root directory of the project. This script will start Minikube, configure the environment, and deploy all services:
``` 
./run.sh

```
- What does the script do:
    
    1. Cluster Setup: Starts Minikube with 4 CPUs and 4GB RAM.

    2. Addons: Enables metrics-server (required for Horizontal Pod Autoscaling).

    3. Security: Automatically generates mongodb-atlas-secret from the .env file.

    4. Deployment: Applies all manifests via Kustomize (kubectl apply -k).

    5. Auto-Access: Opens the Kubernetes Dashboard and the Streamlit UI in the browser once the pods are ready.

- Shutdown & Cleanup
    - To safely spin down the cluster and puase all active deployments, secrets, and configurations, use the code below:

    ```
    ./stop.sh
    ```    

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

