#!/bin/bash

# Directory to clean
DIRECTORY="$(pwd)"

# Check if the directory exists
if [ -d "$DIRECTORY" ]; then
    echo "Deleting files matching pattern *.sh.* in $DIRECTORY"
    # Find and delete files matching the pattern
    find "$DIRECTORY" -type f -name "*.sh.*" -exec rm -v {} +
    echo "Deletion complete."
else
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi
