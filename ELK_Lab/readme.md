# ELK Stack Lab - PitchQuest Pipeline Monitoring

## Overview
This lab demonstrates setting up an ELK (Elasticsearch, Logstash, Kibana) stack for monitoring a machine learning data pipeline. The implementation tracks logs from the PitchQuest multimodal analysis system.

## Setup Instructions

### Prerequisites
- Ubuntu/WSL2
- Java 11+
- Python 3.x

### Installation Steps

1. **Install Elasticsearch**
```bash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
sudo apt update
sudo apt install elasticsearch -y
```

2. **Install Kibana**
```bash
sudo apt install kibana -y
```

3. **Start Services**
```bash
sudo systemctl start elasticsearch
sudo systemctl start kibana
```

4. **Access Kibana**
Open browser: `http://localhost:5601`

## Log Generation

The `generate_pipeline_logs.py` script simulates logs from different pipeline stages:
- video_upload
- audio_extraction  
- transcription
- emotion_detection
- feedback_generation

**Run:**
```bash
python3 generate_pipeline_logs.py
```

## Dashboard

The Kibana dashboard includes:
- Total log count metric
- Pipeline stages distribution (bar chart)
- Logs timeline (area chart)

**Dashboard Name:** PitchQuest Pipeline Monitor

## Files

- `generate_pipeline_logs.py` - Log generation script
- `pitchquest_pipeline.log` - Sample log file
- `README.md` - This file

## Key Learnings

1. ELK stack installation and configuration
2. Log file ingestion into Elasticsearch
3. Creating custom visualizations in Kibana
4. Building monitoring dashboards for ML pipelines

## Result:
The Dashboard is in progress.
<img width="1919" height="856" alt="image" src="https://github.com/user-attachments/assets/98da4d5b-4fc4-47f3-8111-1687b6db448f" />


