# Deploying to Azure

This guide walks through deploying the multimodal application to Azure Container Instances.

## Prerequisites

1. [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed
2. An Azure account with an active subscription
3. Docker installed on your development machine

## Steps to Deploy

### 1. Login to Azure

```bash
az login
```

### 2. Create an Azure Container Registry (ACR)

```bash
# Create a resource group
az group create --name multimodal-resource-group --location eastus

# Create a container registry
az acr create --resource-group multimodal-resource-group --name <your-registry-name> --sku Basic

# Enable admin user
az acr update --name <your-registry-name> --admin-enabled true
```

### 3. Build and Push Docker Image

```bash
# Login to ACR
az acr login --name <your-registry-name>

# Build the Docker image
docker build -t <your-registry-name>.azurecr.io/multimodal-app:latest .

# Push the image to ACR
docker push <your-registry-name>.azurecr.io/multimodal-app:latest
```

### 4. Deploy to Azure Container Instances

Update the `azure-deploy.yaml` file with your:
- Azure region (e.g., eastus, westeurope)
- ACR name

Then deploy:

```bash
# Get ACR credentials
az acr credential show --name <your-registry-name>

# Deploy using the YAML file
az container create --resource-group multimodal-resource-group --file azure-deploy.yaml
```

### 5. Get the Container IP

```bash
az container show --resource-group multimodal-resource-group --name multimodal-app --query ipAddress.ip --output tsv
```

Your application should now be accessible at `http://<container-ip>:8000`

## Troubleshooting

To view container logs:

```bash
az container logs --resource-group multimodal-resource-group --name multimodal-app
```

To restart the container:

```bash
az container restart --resource-group multimodal-resource-group --name multimodal-app
``` 