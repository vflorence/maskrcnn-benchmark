#!/bin/bash
PID=66928
echo "Process: $PID is still running" 
tail --pid=$PID -f /dev/null
python tools/train_net.py --config-file '../../config/SRN_config.yaml' OUTPUT_DIR  './output/SRN_orig' DATASETS.TRAIN '("srn_orig_train",)' DATASETS.TEST '("srn_orig_val",)'
