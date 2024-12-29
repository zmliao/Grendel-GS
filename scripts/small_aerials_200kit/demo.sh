#!/bin/bash
list=("1 2 3" "2 3 4" "3 4 5")
list1=("1 2 3" "2 3 4" "3 4 5")

for i in {0..2}; do
    echo "${list[$i]}"
done