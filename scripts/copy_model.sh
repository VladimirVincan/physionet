#!/bin/bash
# how to run: ./copy_model.sh training_id

local="../physionet/"
echo $1
ssh vladimir.vincan.ivi@10.68.6.58 "cd ~/physionet/scripts; bash download_model.sh $1"
scp -r "vladimir.vincan.ivi@10.68.6.58:~/physionet/scripts/$1/best_model.pt" "$local"
ssh vladimir.vincan.ivi@10.68.6.58 "cd ~/physionet/scripts; bash remove_model.sh $1"
