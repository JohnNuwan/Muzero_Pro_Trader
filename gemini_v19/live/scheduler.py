import time
import schedule
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.live.continuous_learner import ContinuousLearner
from gemini_v19.utils.config import CONTINUOUS_LEARNING_CONFIG

def run_scheduler():
    print("⏰ V19 Scheduler Started")
    
    learner = ContinuousLearner()
    
    def job():
        print(f"⏰ Triggering Nightly Retrain at {datetime.now()}")
        learner.retrain()
        
    # Schedule
    retrain_time = CONTINUOUS_LEARNING_CONFIG['retrain_time']
    schedule.every().day.at(retrain_time).do(job)
    
    print(f"   Scheduled retrain at {retrain_time}")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()
