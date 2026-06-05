#!/bin/bash

ROOT="/home/alex/Programming/taskbased_holodeck/userstudy/static/data"

find "$ROOT" -type f -iname "*.png" | while read -r file; do
    jpg="${file%.png}.jpg"
        echo "Converting: $file -> $jpg"
	    convert "$file" "$jpg"
    done

    echo "Done."
