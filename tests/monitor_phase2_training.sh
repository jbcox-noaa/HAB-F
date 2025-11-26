#!/bin/bash
# Monitor Phase 2 training progress

LOG_FILE="mc_lstm_forecasting/training_dual_channel_v2_250epochs.log"

echo "=========================================="
echo "Phase 2 Training Monitor"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo ""

# Check if training is running
if ps aux | grep -q "[p]ython.*train_dual_channel_v2"; then
    echo "✓ Training is RUNNING"
    PID=$(ps aux | grep "[p]ython.*train_dual_channel_v2" | awk '{print $2}')
    echo "  PID: $PID"
else
    echo "✗ Training is NOT running"
fi

echo ""
echo "=========================================="
echo "Recent Progress (last 20 epochs)"
echo "=========================================="

# Extract recent epochs and best validation losses
tail -200 "$LOG_FILE" | grep -E "(Epoch [0-9]+/250|val_loss improved|val_loss did not improve|early stopping)" | tail -20

echo ""
echo "=========================================="
echo "Best Validation Loss So Far"
echo "=========================================="

# Find the best validation loss
grep "val_loss improved" "$LOG_FILE" | tail -1 | grep -oE "val_loss improved from [0-9.]+ to [0-9.]+" || echo "No improvements logged yet"

echo ""
echo "=========================================="
echo "Training Summary"
echo "=========================================="

# Count total epochs completed
COMPLETED=$(grep -c "Epoch [0-9]\+/250" "$LOG_FILE")
echo "Epochs completed: $COMPLETED / 250"

# Check if training finished
if grep -q "STEP 7: EVALUATION" "$LOG_FILE"; then
    echo "Status: ✓ COMPLETE"
    echo ""
    echo "Final Results:"
    tail -100 "$LOG_FILE" | grep -A 10 "Test Results:"
elif grep -q "early stopping" "$LOG_FILE"; then
    echo "Status: ✓ STOPPED EARLY"
    STOP_EPOCH=$(grep "early stopping" "$LOG_FILE" | grep -oE "Epoch [0-9]+" | tail -1)
    echo "Stopped at: $STOP_EPOCH"
else
    echo "Status: → IN PROGRESS"
    
    # Estimate completion time (rough estimate: 9 seconds per epoch)
    REMAINING=$((250 - COMPLETED))
    SECONDS=$((REMAINING * 9))
    MINUTES=$((SECONDS / 60))
    echo "Estimated remaining: ~$MINUTES minutes ($REMAINING epochs)"
fi

echo ""
echo "=========================================="
echo "To view live updates, run:"
echo "  tail -f $LOG_FILE | grep -E 'Epoch|val_loss'"
echo "=========================================="
