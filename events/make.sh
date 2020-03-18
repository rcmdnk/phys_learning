#!/usr/bin/env bash

name=$1
proc=$2


card=$(mktemp)
cat <<EOF > $card
import model sm
define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~
define l+ = e+ mu+
define l- = e- mu-
define vl = ve vm vt
define vl~ = ve~ vm~ vt~

generate $proc

output $name
EOF


python2 ./bin/mg5_aMC $card
rm -f $card

cd $name
python2 ./bin/generate_events -f
gunzip ./Events/run_01/unweighted_events.lhe.gz
cd ../
