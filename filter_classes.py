import sys

with open(sys.argv[1]) as fin:
    klass = 0
    words = []
    for line in fin:
        if line == '\n':
            if len(words) >= 2:
                sys.stdout.write('Class {}\n'.format(klass) + ''.join(words) + '\n')
                klass += 1
            words = []
        elif line[:5] == 'Class':
            pass
        else:
            split = line.strip().split()
            if float(split[2]) - float(split[1]) >= 0.5:
                words.append(line)
            
