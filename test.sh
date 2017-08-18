#!/bin/bash

for ((i=0; i<=1000; i++)); do
    /home/sauhaarda/anaconda3/bin/python Train.py --epoch $i "$@" 
done
