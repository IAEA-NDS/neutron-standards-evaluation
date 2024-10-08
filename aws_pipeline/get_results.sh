#!/bin/bash

branch_name="$1"
output_dir="../output"
branch_output_dir="$output_dir/$branch_name"

last_output_file="03_mcmc_sampling_output.pkl"

while true; do
    echo Checking if results for on branch "$branch_name" are already available...
    aws s3 ls "s3://gmapy-results/$branch_name/" --recursive | grep "$last_output_file"
    if [ $? -eq 0 ]; then
        break
    fi
    echo Waiting for results on branch "$branch_name"...
    sleep 120
done

mkdir -p "$branch_output_dir"

aws s3 cp "s3://gmapy-results/$branch_name/" "$branch_output_dir"    --recursive
