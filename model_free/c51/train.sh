#!/bin/bash

for env in {1..3}; do
    for i in {1..3}; do
        if [ $env -eq 1 ]; then
            echo "Running PongNoFrameskip-v1, iteration $i"
            python main.py --env $env
        elif [ $env -eq 2 ]; then
            echo "Running BreakoutNoFrameskip-v4 iteration $i"
            python main.py --env $env
        else
            echo "Running SeaquestNoFrameskip-v4 iteration $i"
            python main.py --env $env
        fi
        python main.py --env $env
    done
done

