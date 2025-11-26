"""
Quick test evaluation of trained MC forecasting model.
"""
import numpy as np
from mc_lstm_forecasting.utils import load_mc_sequences, configure_logging
from mc_lstm_forecasting.model import build_mc_convlstm_model

configure_logging()

print("Loading test data...")
X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_mc_sequences()

# Replace NaN with 0
X_test = np.nan_to_num(X_test, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

print(f"Test set: {len(X_test)} sequences")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

print("\nBuilding model architecture...")
model = build_mc_convlstm_model(input_shape=(5, 84, 73, 1))

print("Loading trained weights from best_model.keras...")
model.load_weights('mc_lstm_forecasting/best_model.keras')
print("✅ Weights loaded successfully!")

print("\nEvaluating on test set (2025 Aug-Oct peak bloom)...")
test_results = model.evaluate(X_test, y_test, batch_size=16, verbose=1)

print("\n" + "="*60)
print("TEST SET RESULTS (2025 Aug-Oct Peak Bloom)")
print("="*60)
print(f"Loss (MSE):           {test_results[0]:.6f}")
print(f"MAE:                  {test_results[1]:.6f}")
print(f"MSE:                  {test_results[2]:.6f}")
print(f"RMSE:                 {np.sqrt(test_results[2]):.6f}")
print("="*60)
