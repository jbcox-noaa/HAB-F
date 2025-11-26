#!/usr/bin/env python
"""Monitor MC forecasting training progress."""

import time
import os
from pathlib import Path

log_file = Path("mc_lstm_forecasting/training_log.txt")

print("Monitoring MC Forecasting Training...")
print("=" * 80)

last_size = 0
no_change_count = 0

try:
    while True:
        if log_file.exists():
            current_size = log_file.stat().st_size
            
            if current_size > last_size:
                # Read new content
                with open(log_file, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    
                    # Print only lines with "Epoch" or important info
                    for line in new_content.split('\n'):
                        if any(keyword in line for keyword in [
                            'Epoch ', '/100', 'val_loss:', 'Best model', 
                            'COMPLETE', 'Test', 'Early stopping'
                        ]):
                            print(line)
                    
                last_size = current_size
                no_change_count = 0
            else:
                no_change_count += 1
                
                # If no changes for 30 seconds, check if process is done
                if no_change_count > 6:
                    print("\n✓ Training appears to be complete or paused.")
                    print("Check mc_lstm_forecasting/training_log.txt for full details.")
                    break
        
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user.")
    print("Training is still running in background.")
    print("Check mc_lstm_forecasting/training_log.txt for progress.")
