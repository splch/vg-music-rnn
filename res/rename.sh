a=1
for i in *.mid; do
  new=$(printf "%05d.midi" "$a") #05 pad to length of 5
  mv -i -- "$i" "$new"
  let a=a+1
done