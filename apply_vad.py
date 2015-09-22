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

# with open(disc_file) as fin:
#     for line in fin:
#         if line == '\n' or line[:5] == 'Class':
#             sys.stdout.write(line)
#         else:
#             fname, start, end = line.split()
#             start, end = float(start), float(end)
#             if any(start > x[0] and end < x[1] for x in vad[fname]):
#                 sys.stdout.write(line)

with open(disc_file) as fin:
    for line in fin:
        if line[0:4] == 'file' or line[0:4] == 'fnam':
            continue
        else:
            try:
                fname1, fname2, start1, end1, start2, end2 = line.split()[:6]
            except ValueError:
                fname1, fname2, start1, end1, start2, end2 = line.split(',')[:6]                
            start1, end1, start2, end2 = map(lambda x: float(x)/100, [start1, end1, start2, end2])
            if (fname1 == 's0103a' or fname2 == 's0103a'):
                continue
            if any(start1 > x[0] and end1 < x[1] for x in vad[fname1]):
                if any(start2 > x[0] and end2 < x[1] for x in vad[fname2]):
                    sys.stdout.write(line)
