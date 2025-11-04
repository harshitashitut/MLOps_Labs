"""
Custom log generator for PitchQuest Data Pipeline
Simulates logs from video processing, transcription, and model inference
"""
import logging
import random
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='pitchquest_pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('PitchQuest-Pipeline')

# Simulate different pipeline stages
stages = ['video_upload', 'audio_extraction', 'transcription', 'emotion_detection', 'feedback_generation']
statuses = ['success', 'warning', 'error']

print("Generating PitchQuest pipeline logs...")

for i in range(50):
    stage = random.choice(stages)
    
    # 70% success, 20% warning, 10% error
    status = random.choices(statuses, weights=[70, 20, 10])[0]
    
    if status == 'success':
        logger.info(f"Stage '{stage}' completed successfully (video_{i+1})")
    elif status == 'warning':
        logger.warning(f"Stage '{stage}' completed with warnings (video_{i+1}) - slow processing time")
    else:
        logger.error(f"Stage '{stage}' failed (video_{i+1}) - timeout or model error")
    
    time.sleep(0.1)  # Small delay

print("✅ Generated 50 log entries in 'pitchquest_pipeline.log'")