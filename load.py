"""Module for loading gold transcription and spoken terms
and annotate them

Gold trans:
file
onset offset phone
onset offset phone
...
(time in sec)

spoken terms:
Class classid
file onset offset
file onset offset
...

Class classid
...
"""
import tde.data.interval as interval
import os, errno
import numpy as np
from recordtype import recordtype
import itertools


# RESOLUTION PARAMETERS:
FRATE = 100  # number of frames per seconds

SIL_LABEL = 'SIL'

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            # errno.ENOENT = no such file or directory
            raise

LabelledInterval = recordtype('LabelledInterval' ,'label interval')


def load_gold(goldfile):
    res = {}
    current_file = ''
    prev_end = -1
    with open(goldfile) as fin:
        for line in fin:
            splitted = line.strip().split()
            if not splitted:
                # empty line
                continue
            elif len(splitted) == 1:
                # New file
                current_file = splitted[0]
                res[current_file] = []
                prev_end = 0
            else:
                assert len(splitted) == 3, splitted
                start, end = map(float, splitted[:2])
                if prev_end != start:
                    astart, aend = prev_end, start
                    res[current_file].append(LabelledInterval(
                        SIL_LABEL, interval.Interval(astart, aend)))
                phone = splitted[2]
                res[current_file].append(LabelledInterval(
                    phone, interval.Interval(start, end)))
                prev_end = end
    return res


# def load_pos(buckeye_root):
#     """Load the POS tagging from the buckeye corpus

#     The corpus should be kept in the standard format
#     (as on 23/07/2015 on oberon)
#     """
#     import glob
#     res = {}
#     for f in glob.glob(os.path.join(buckeye_root, '*', '*', '*.word')):
#         fname = os.path.basename(f).split('.')[0]
#         res[fname] = []
#         with open(f) as fin:
#             while fin.readline() != '#\n':
#                 pass
#             token = fin.readline().strip().replace(';', '').split()


def load_disc(classfile):
    res = {}
    current_class = ''
    with open(classfile) as fin:
        for line in fin:
            splitted = line.strip().split()
            if not splitted:
                # empty line
                if current_class and len(res[current_class]) == 1:
                    del res[current_class]
            elif splitted[0] == 'Class':
                # New class
                current_class = splitted[1]
                res[current_class] = []
            else:
                assert len(splitted) == 3, splitted
                start, end = map(float, splitted[1:])
                fname = splitted[0]
                res[current_class].append(TranscribedInterval(
                    fname, interval.Interval(start, end), None, None, {}))
        if len(res[current_class]) == 1:
            del res[current_class]
    return res


def load_disc_pairs(pairfile, times='seconds'):
    res = {}
    current_class = ''
    with open(pairfile) as fin:
        for current_class, line in enumerate(fin):
            if line[0:4] == 'file' or line[0:4] == 'fnam':
                continue
            res[current_class] = []
            splitted = line.strip().split(',')
            if len(splitted) == 1:
                splitted = line.strip().split()
            # assert len(splitted) == 8, splitted
            if times == 'seconds':
                start1, end1, start2, end2 = map(float, splitted[2:6])
            else:
                start1, end1, start2, end2 = map(lambda x: float(x)/100, splitted[2:6])
            fname1, fname2 = splitted[0:2]
            if len(splitted) > 6:
                dtw_aren, score_aren = map(float, splitted[6:])
            else:
                dtw_aren, score_aren = (0., 0.)
            res[current_class].append(TranscribedInterval(
                fname1, interval.Interval(start1, end1), None, None,
                #['None'], ['None'] * int((end1 - start1) * 100),
                {'dtw_aren': dtw_aren, 'score_aren': score_aren}))
            res[current_class].append(TranscribedInterval(
                fname2, interval.Interval(start2, end2), None, None,
                # ['None'], ['None'] * int((end2 - start2) * 100),
                {'dtw_aren': dtw_aren, 'score_aren': score_aren}))
    return res


TranscribedInterval = recordtype(
    'TranscribedInterval',
    'fname interval phone_transcription'
    ' frame_transcription stats')


def sort_disc(disc_dict):
    """Return a dictionnary by file of the interval sorted by start
    """
    # iterator over all intervals:
    # def iter_all(dict_of_lists):
    #     for l in dict_of_lists.itervalues():
    #         for e in l:
    #             yield e
    it_all = itertools.chain.from_iterable(disc_dict.itervalues())
    return {fname: sorted(interv_it, key=lambda x: x.interval.start)
           for fname, interv_it in itertools.groupby(
                   sorted(it_all, key=lambda x: x.fname),
                   key=lambda x: x.fname)}


def annotate_phone_disc(gold_dict, disc_dict, disc_list=None):
    """Annotate discovered fragments with gold at phone level.
    Change in place the (mutable) discovered fragments"""
    for fname in disc_dict:
        phone_index = 0
        last_silence = 0.  # end of last encountered silence
        current_last_silence = 0.
        for disc_interval in disc_dict[fname]:
            transcription = []
            fname = disc_interval.fname
            transcription_started = False
            transcription_intervals = []
            # reading the corresponding gold transcription
            curr_phone_index = phone_index
            start_phone_index = phone_index
            while curr_phone_index < len(gold_dict[fname]):
            # for phone_interval in gold_dict[fname][aux:]:
                phone_interval = gold_dict[fname][curr_phone_index]
                if phone_interval.interval.overlaps_with(disc_interval.interval):
                    if not transcription_started: 
                       current_last_silence = last_silence
                    transcription_started = True
                    transcription.append(phone_interval.label)
                    transcription_intervals.append(phone_interval)
                elif transcription_started:
                    # transcription started, but phone non overlap =>
                    # end of transcription
                    break
                else:
                    # transcription not started, first phone not encontered
                    # let's increment our phone counter
                    phone_index += 1
                if phone_interval.label == SIL_LABEL:
                    last_silence = phone_interval.interval.end
                curr_phone_index += 1
            disc_interval.stats['prev_sil'] = max(
                0, disc_interval.interval.start - current_last_silence)
            # finding next silence:
            disc_interval.stats['next_sil'] = max(
                0, gold_dict[fname][-1].interval.start -
                disc_interval.interval.end)
            if not transcription_intervals:
                transcription_intervals = ['None']
                print "Error:"
                print disc_interval
                phone_index = start_phone_index
            frame_transcription = frame_alignement(
                transcription_intervals, disc_interval.interval)
            disc_interval.phone_transcription = transcription
            disc_interval.frame_transcription = frame_transcription
            for phone_interval in gold_dict[fname][
                    phone_index+len(disc_interval.phone_transcription)-1:]:
                if phone_interval.label == SIL_LABEL:
                    disc_interval.stats['next_sil'] = max(
                        0, phone_interval.interval.start -
                        disc_interval.interval.end)
                    break



def frame_alignement(interval_list, disc_interval):
    phones = [intervl.label for intervl in interval_list]
    onsets = np.array([intervl.interval.start for intervl in interval_list])
    try:
        onsets[0]
    except:
        print interval_list, disc_interval
        raise
    onsets[0] = disc_interval.start
    offsets = np.array([intervl.interval.end for intervl in interval_list])
    offsets[-1] = disc_interval.end
    nframes = np.around((offsets - onsets) * FRATE)
    res = [[phone] * nframe for phone, nframe in zip(phones, nframes)]
    res = [item for sublist in res for item in sublist]
    return res


def tests():
    """tests"""
    import tempfile
    try:
        _, gold_file = tempfile.mkstemp()
        _, disc_file = tempfile.mkstemp()

        gold_text = """
file1
0.0 0.5 a
0.5 1.0 b
1.0 1.5 c
1.5 2.0 d

file2
0.0 0.5 e
0.5 1.0 f
1.0 1.5 g
1.5 2.0 h"""
        disc_text = """
Class 1
file1 0.1 1.4
file1 0.6 1.9

Class 2
file1 0.2 1.4
file2 0.7 1.9"""
        def write_to_file(path, text):
            with open(path, 'w') as fin:
                fin.write(text)
        map(write_to_file, (gold_file, disc_file), (gold_text, disc_text))
        gold_phones = load_gold(gold_file)
        disc_frags = load_disc(disc_file)
        sort_disc(disc_frags)
        return annotate_phone_disc(gold_phones, disc_frags)
    finally:
        silentremove(gold_file)
        silentremove(disc_file)
