#!/bin/bash
FOLDER="data_generation"
for branch in $(git branch -a | sed 's/\*//g' | tr -d ' '); do
    echo "Checking branch: $branch"
    if git ls-tree -d $branch -- $FOLDER 2>/dev/null | grep -q "$FOLDER"; then
        echo "Folder '$FOLDER' exists in branch $branch"
    else
        echo "Folder '$FOLDER' does not exist in branch $branch"
    fi
done
