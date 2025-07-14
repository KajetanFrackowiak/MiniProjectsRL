#!/bin/bash

METHOD=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

for env in {1..3}; do
    for i in {1..3}; do
        if [ $env -eq 1 ]; then
            python main.py --env $env --method $METHOD
        elif [ $env -eq 2 ]; then
            python main.py --env $env --method $METHOD
        elif [ $env -eq 3 ]; then
            python main.py --env $env --method $METHOD
        else
            echo "Invalid environment number: $env"
            exit 1
        fi
    done
done
        