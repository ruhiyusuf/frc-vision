#!/bin/bash
echo "Press [CTRL+C] to stop..."
( 
while : 
do
 echo "$(date +'%S')" | nc -u -w0 localhost 5000
 sleep 1 
done
)
