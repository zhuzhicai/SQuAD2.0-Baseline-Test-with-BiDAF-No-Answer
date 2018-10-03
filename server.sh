#!/bin/bash
#Save the current server command
set -eu -o pipefail
python server/demo.py tmpout -d -t ../neg-questions/out/0xa41dd2/out
