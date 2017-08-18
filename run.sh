#!/bin/bash

for ((i=0; i<=1000; i++)); do
    tpankaj-docker python Train.py --epoch $i "$@" 
done
