#!/usr/bin/env bash

set -exo pipefail

# Download dataset.
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip -d data MPI-Sintel-complete.zip

# Create directory structure for outputs.
rsync -a -f"+ */" -f"- *" data/training/ data/output
