"""
Training Options
"""
import os

# layer_size: 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
LAYER_SIZE = 8 # default
# stack_size: 5 = stack[layer1, layer2, layer3, layer4, layer5]
STACK_SIZE = 5
# input channel size. mu-law encode factor, one-hot size
IN_CHANNELS=256
# number of channel for residual network
RES_CHANNELS=512
# lr
LEARNING_RATE = 0.0002
# Training data dir
DATA_DIR = './test/data' 
# Total training steps
NUM_STEPS = 10000

TEST_OUTPUT_DIR = 'wavenet/test_output'
OUTPUT_DIR = 'wavenet/output'

LOG_DIR = os.path.join(OUTPUT_DIR, 'log')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'test')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)