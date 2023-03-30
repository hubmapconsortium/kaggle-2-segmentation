#!/bin/bash
echo "$1"
if [ "$1" = "val" ]
then
    echo "running val pred"
    python predict_val.py
elif [ "$1" = "external" ]
then 
    echo "running ext pred"
    python predict_external.py
elif [ "$1" = "normal" ]
then
    echo "running normal pred"
    python predict.py
else
    echo "incorrect parameter"
fi
