#! /bin/bash

output_name=$1


echo "$2" | tr -s ';' ' ' | xargs ar -crs $output_name
