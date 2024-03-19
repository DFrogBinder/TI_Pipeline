#!/usr/bin/env bash

set -e

cd '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Export_Save_Dir'

for file in *.poly; do 
    if [ -f "$file" ]; then 
        time '/usr/bin/tetgen' -zpq1.8/0O4a30kNEFAV "$file"
    fi 
done
