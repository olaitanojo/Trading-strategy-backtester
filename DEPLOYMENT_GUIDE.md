# 🚀 Trading Strategy Backtester - Deployment Guide

Comprehensive guide for deploying the Trading Strategy Backtester in various environments.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Setup](#production-setup)
6. [Configuration](#configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)

---

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+ or Docker
- Git
- 4GB+ RAM recommended
- 10GB+ disk space for data and results

### 1-Minute Setup
```bash
# Clone the repository
git clone https://github.com/olaitanojo/trading-strategy-backtester.git
cd trading-strategy-backtester

# Run with Docker (recommended)
docker-compose up -d

# Or run locally
pip install -r requirements.txt
python backtester.py
```

---

## 💻 Local Development Setup

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv trading_env

# Activate environment
# On Windows:
trading_env\Scripts\activate
# On macOS/Linux:
source trading_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration
```bash
# Copy and modify configuration
cp config.yaml config_local.yaml
# Edit config_local.yaml with your preferences
```

### Step 3: Run Tests
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Step 4: Start Backtesting
```bash
# Run main backtester
python backtester.py

# Run specific modules
python technical_indicators.py
python advanced_strategies.py
python risk_management.py
python portfolio_optimization.py
python walk_forward_analysis.py
```

---

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t trading-backtester .

# Run container
docker run -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  trading-backtester
```

### Multi-Service with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale backtester service
docker-compose up -d --scale backtester=3

# Stop services
docker-compose down
```

### Services Available:
- **Backtester**: Main trading analysis engine
- **Jupyter**: Interactive notebooks at `http://localhost:8888`
- **PostgreSQL**: Database for storing results
- **Redis**: Caching layer for improved performance

---

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: EC2 Instance
```bash
# Launch EC2 instance (t3.large recommended)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# Deploy application
git clone https://github.com/olaitanojo/trading-strategy-backtester.git
cd trading-strategy-backtester
sudo docker-compose up -d

# Setup security group (ports: 22, 8888)
```

#### Option 2: ECS with Fargate
```yaml
# ecs-task-definition.json
{
  "family": "trading-backtester",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backtester",
      "image": "your-ecr-repo/trading-backtester:latest",
      "portMappings": [
        {
          "containerPort": 8888,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trading-backtester",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Build and push to Container Registry
docker build -t gcr.io/PROJECT_ID/trading-backtester .
docker push gcr.io/PROJECT_ID/trading-backtester

# Deploy to Cloud Run
gcloud run deploy trading-backtester \
  --image gcr.io/PROJECT_ID/trading-backtester \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Microsoft Azure

#### Container Instances
```bash
# Create resource group
az group create --name trading-rg --location eastus

# Create container instance
az container create \
  --resource-group trading-rg \
  --name trading-backtester \
  --image your-registry/trading-backtester:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8888 \
  --dns-name-label trading-backtester-unique
```

---

## 🏭 Production Setup

### High Availability Configuration

#### Load Balancer Setup (nginx)
```nginx
upstream backtester {
    server backtester1:8000;
    server backtester2:8000;
    server backtester3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://backtester;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Auto-scaling Docker Compose
```yaml
version: '3.8'
services:
  backtester:
    build: .
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

### Database Setup

#### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE trading_db;
CREATE USER trader WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trader;

-- Create tables for results
\c trading_db;
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parameters JSONB,
    performance_metrics JSONB,
    equity_curve JSONB
);

CREATE INDEX idx_strategy_timestamp ON backtest_results(strategy_name, timestamp);
```

### Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'trading-backtester'
    static_configs:
      - targets: ['backtester:9090']
```

#### Grafana Dashboard
Import dashboard JSON for trading metrics visualization.

---

## ⚙️ Configuration

### Environment Variables
```bash
# Data settings
export TRADING_DATA_SOURCE=yfinance
export TRADING_CACHE_ENABLED=true

# Risk management
export TRADING_MAX_RISK_PER_TRADE=0.02
export TRADING_INITIAL_CAPITAL=100000

# Database
export DATABASE_URL=postgresql://user:pass@host:port/db
export REDIS_URL=redis://localhost:6379

# API keys
export ALPHA_VANTAGE_API_KEY=your_key_here
export IEX_TOKEN=your_token_here
```

### Configuration Management
```python
# config_manager.py
import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"):
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith('TRADING_'):
                config_key = key[8:].lower()
                config[config_key] = value
        
        return config
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
```

---

## 📊 Monitoring & Logging

### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                'backtester.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
```

### Health Check Endpoint
```python
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    })
```

### Alerts Configuration
```yaml
# alerts.yml (Prometheus)
groups:
  - name: trading-backtester
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          description: "CPU usage is above 80%"
      
      - alert: BacktestFailed
        expr: increase(backtest_errors_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          description: "Backtest execution failed"
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limit
docker run --memory=4g trading-backtester

# Optimize pandas operations
export PANDAS_MEMORY_EFFICIENT=true
```

#### 2. Data Download Failures
```python
# Retry mechanism for yfinance
import time
import yfinance as yf

def robust_download(ticker, **kwargs):
    for attempt in range(3):
        try:
            return yf.download(ticker, **kwargs)
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2 ** attempt)
```

#### 3. Performance Issues
```bash
# Profile code
python -m cProfile -o profile.stats backtester.py

# Use faster data structures
pip install modin[ray]  # Faster pandas operations
```

#### 4. Container Issues
```bash
# Debug container
docker run -it --entrypoint /bin/bash trading-backtester

# Check logs
docker logs trading-backtester

# Clean up
docker system prune -a
```

### Debugging Commands
```bash
# Check service status
docker-compose ps

# View real-time logs
docker-compose logs -f backtester

# Execute commands in container
docker-compose exec backtester python -c "import pandas; print(pandas.__version__)"

# Restart specific service
docker-compose restart backtester

# Scale services
docker-compose up -d --scale backtester=2
```

---

## 🆘 Support

### Getting Help
- 📧 Email: support@trading-backtester.com
- 💬 Discord: https://discord.gg/trading-backtester
- 📋 Issues: https://github.com/olaitanojo/trading-strategy-backtester/issues
- 📖 Wiki: https://github.com/olaitanojo/trading-strategy-backtester/wiki

### Performance Benchmarks
- **Small dataset** (1 year, 1 strategy): ~30 seconds
- **Medium dataset** (3 years, 5 strategies): ~2-3 minutes
- **Large dataset** (10 years, 20 strategies): ~10-15 minutes
- **Walk-forward analysis**: ~30-60 minutes

### System Requirements

#### Minimum
- 2 CPU cores
- 4GB RAM
- 5GB disk space
- Python 3.8+

#### Recommended
- 4+ CPU cores
- 8GB+ RAM
- 20GB+ SSD storage
- Python 3.9+

#### Production
- 8+ CPU cores
- 16GB+ RAM
- 100GB+ SSD storage
- Load balancer
- Database cluster
- Monitoring stack

---

*Last updated: December 2024*
*Version: 2.0.0*
