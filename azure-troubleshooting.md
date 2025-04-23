# Azure Deployment Troubleshooting Guide

## Common Issues and Solutions

### 1. Website Not Loading

#### Check Application Logs
```bash
az webapp log tail --name <your-app-name> --resource-group <your-resource-group>
```

#### Check Deployment Status
- Go to Azure Portal > Your Web App > Deployment Center
- Check if deployment completed successfully
- Look for any error messages in the deployment logs

#### Check Application Settings
- Verify PORT is set to 8000 (matches your Dockerfile and app.py)
- In Azure Portal > Your Web App > Configuration > Application settings
- Add setting: `WEBSITES_PORT = 8000`

### 2. Container Issues

#### Check Container Logs
- Azure Portal > Your Web App > Container settings
- Check for any startup errors

#### Restart the Web App
- Azure Portal > Your Web App > Overview > Restart
- Wait a minute and try accessing again

### 3. Networking Issues

#### Check if Web App is Running
- Try accessing the /health endpoint: https://<your-app-name>.azurewebsites.net/health
- If this works but the main page doesn't, check your app's routing

#### CORS Issues
- If using API calls from a frontend, check CORS settings

### 4. Docker Configuration

#### Verify Dockerfile
- Make sure the Dockerfile exposes port 8000
- Ensure gunicorn is binding to 0.0.0.0:8000
- Check that requirements.txt includes all dependencies

#### Manual Container Test (Local)
```bash
docker build -t multimodal-app .
docker run -p 8000:8000 multimodal-app
```

### 5. Application Code Issues

#### Check for Hardcoded URLs
- Make sure you're not using hardcoded localhost URLs
- Replace with relative paths or environment variables

#### Environment Variables
- Make sure any required environment variables are set in Azure

### 6. Force Redeployment
- Make a small change to your code (like a comment)
- Commit and push to GitHub
- This will trigger a new deployment

### 7. Scale Up Resources
- If the app is slow or timing out, consider scaling up your App Service Plan

## Quick Fix Checklist

1. Added /health endpoint to verify app is running
2. Changed port from 5000 to 8000 in app.py
3. Ensured Dockerfile exposes port 8000
4. Set WEBSITES_PORT=8000 in Azure App Settings
5. Restarted the Web App
6. Checked application logs for specific errors 