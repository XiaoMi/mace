#! /bin/bash

output_name=$1

shift

for i in "$@"
do
    echo "$i" | tr -s ';' ' ' | xargs ar qc $output_name
done