#!/usr/bin/env bash

wget https://launchpad.net/mg5amcnlo/2.0/2.7.x/+download/MG5_aMC_v2.7.2.tar.gz
tar zxf MG5_aMC_v2.7.2.tar.gz
rm -f MG5_aMC_v2.7.2.tar.gz

cd MG5_aMC_v2_7_2
../make.sh my_z "p p > z > j j"
../make.sh my_jj "p p > j j"
cd ../

./lhe2txt.py MG5_aMC_v2_7_2/my_z/Events/run_01/unweighted_events.lhe > my_z.txt
./lhe2txt.py MG5_aMC_v2_7_2/my_jj/Events/run_01/unweighted_events.lhe > my_jj.txt

while read line;do echo $line 1;done < my_z.txt > data.txt
while read line;do echo $line 0;done < my_jj.txt >> data.txt
