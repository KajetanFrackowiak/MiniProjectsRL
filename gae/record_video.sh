#!/bin/bash
# Easy video recording script for RL evaluation

# Set environment variables for MuJoCo rendering
export MUJOCO_GL=osmesa

# Pass all arguments to the evaluate script
python evaluate.py "$@"
