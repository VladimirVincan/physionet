#!/bin/bash

./../login.sh
fmle-cli e t download $1
unzip $1.zip
rm -r $1*
