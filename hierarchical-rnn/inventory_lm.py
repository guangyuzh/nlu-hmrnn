"""
Configurations for character-level language modeling
"""

# Prepare inputs
BATCH_SIZE = 20
NUM_BATCHES = 1500
TRUNCATE_LEN = 1000
STEP_SIZE = 500

# for HMLSTMNetwork
OUTPUT_SIZE = 27
INPUT_SIZE = 27
EMBED_SIZE = 2048
OUT_HIDDEN_SIZE = 1024
HIDDEN_STATE_SIZES = 1024
