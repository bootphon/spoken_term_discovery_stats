import sys
vadfile = 'test/english_vad'
disc_file = sys.argv[1]

vad = {}
with open(vadfile) as fin:
    fin.readline()
    for line in fin:
        fname, start, end = line.split(',')
        if fname not in vad:
            vad[fname] = []
        vad[fname].append([float(start), float(end)])

with open(disc_file) as fin:
    for line in fin:
        if line == '\n' or line[:5] == 'Class':
            sys.stdout.write(line)
        else:
            fname, start, end = line.split()
            start, end = float(start), float(end)
            if any(start > x[0] and end < x[1] for x in vad[fname]):
                sys.stdout.write(line)
