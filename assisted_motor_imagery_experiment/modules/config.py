#Classifier Parameters
FS = 250
NUM_CHANNELS = 8

FREQ_LOW = 8
FREQ_HIGH = 30
FFT_WINDOW_SIZE_SEC = 0.25
FFT_WINDOW_SIZE_SAMPLES = int(FFT_WINDOW_SIZE_SEC * FS)
FFT_STEP_SIZE_SEC = 1.0/FS
FFT_STEP_SIZE_SAMPLES = int(FFT_STEP_SIZE_SEC * FS)
NFFT = 256
CLASSIFICATION_WINDOW_SEC = 0.5
CLASSIFICATION_WINDOW_SAMPLES = int(CLASSIFICATION_WINDOW_SEC * FS)
CLASSIFICATION_WINDOW_STEP_SEC = 1.0/FS
# CLASSIFICATION_WINDOW_STEP_SEC = 1.0
CLASSIFICATION_WINDOW_STEP_SAMPLES = int(CLASSIFICATION_WINDOW_STEP_SEC * FS)
CLASSIFICATION_MARKER_TIME_THRESHOLD_SEC = 0.5
CLASSIFICATION_MARKER_TIME_THRESHOLD_SAMPLES = int(CLASSIFICATION_MARKER_TIME_THRESHOLD_SEC * FS)
CLASSIFICATION_FILTERING_PADDING_SEC = 0.2
CLASSIFICATION_FILTERING_PADDING_SAMPLES = int(CLASSIFICATION_FILTERING_PADDING_SEC * FS)

#OpenBCI Definitions
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"

SAMPLE_RATE = 250  # Hz
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

# Motor Imagery Experiment Parameters
NUM_TRIALS = 102  # Configurable number of trials
READY_DURATION = 1.0  # seconds to show "Get Ready" message
CUE_DURATION = 0.25  # seconds to show START/STOP cue
INSTRUCTION_DURATION = 2.0  # seconds to show left/right instruction
IMAGERY_DURATION = 3.0  # seconds for motor imagery
INTER_TRIAL_INTERVAL = 3.0  # seconds between trials
#Whole trial is 9 seconds

# Marker values
MARKER_RIGHT = "1"  # right arm imagery
MARKER_LEFT = "2"   # left arm imagery
MARKER_LEG = "3"   # leg imagery
MARKER_STOP = "4"   # end of imagery period
MARKER_FAIL = "5"   # failed to add marker

# File/directories
RAW_DATA_DIR = "./data/raw"

# Data collection parameters
MERGE_THRESHOLD = 0.002  # seconds threshold for aligning EEG and marker timestamps
POLL_SLEEP = 0.001      # sleep time between polls in collector loop