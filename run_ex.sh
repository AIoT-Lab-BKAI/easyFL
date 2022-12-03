#!/bin/bash

read -p 'Enter the directory path: ' directory
for file in "$directory"/*; do
  CUDA_VISIBLE_DEVICES=0,1
  bash "$file"
done