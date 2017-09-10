#!/bin/bash
FILES=*.mid
for f in $FILES
do
  echo "Converting $f to MP3..."
  timidity $f -Ow -o - | lame - -b 64 $f.mp3
done
