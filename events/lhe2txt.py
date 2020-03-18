#!/usr/bin/env python


import sys
fname = sys.argv[1]
with open(fname) as f:
    lines = f.readlines()

info = ""
flag = 0
for line in lines:
    line = line.strip()
    if flag == 0:
        if line != "<event>":
            continue
        flag = 1
    elif flag in (1, 2, 3):
        flag += 1
    elif line == "<mgrwt>":
        print(info)
        info = ""
        flag = 0
    else:
        pid = abs(int(line.split()[0]))
        if pid not in (1, 2, 3, 4, 21):
            continue
        info += ' ' + ' '.join(line.split()[6:10])
