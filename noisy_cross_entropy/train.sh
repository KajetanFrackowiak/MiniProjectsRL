#!/bin/bash

for env in CartPole-v1 Pendulum-v1 MountainCar-v0 Acrobot-v1 LunarLander-v3; do
  for i in {1..3}; do
    echo "Running $env, iteration $i"
    python main.py --method Noisy_CrossEntropy --env $env
  done
done

