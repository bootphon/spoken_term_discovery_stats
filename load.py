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


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

LabelledInterval = recordtype('LabelledInterval' ,'label interval')


def load_gold(goldfile):
    res = {}
    current_file = ''
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
            else:
                assert len(splitted) == 3, splitted
                start, end = map(float, splitted[:2])
                phone = splitted[2]
                res[current_file].append(LabelledInterval(
                    phone, interval.Interval(start, end)))
    return res


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
                    fname, interval.Interval(start, end), None, None))
        if len(res[current_class]) == 1:
            del res[current_class]
    return res


TranscribedInterval = recordtype('TranscribedInterval',
                                 'fname interval phone_transcription'
                                 ' frame_transcription')


def sort_disc(disc_dict):
    """Return a dictionnary by file of the interval sorted by start
    """
    # iterator over all intervals:
    it_all = itertools.chain.from_iterable(disc_dict.itervalues())
    return {fname: sorted(interv_it, key=lambda x: x.interval.start)
           for fname, interv_it in itertools.groupby(
                   it_all,
                   key=lambda x: x.fname)}


def annotate_phone_disc(gold_dict, disc_dict):
    """Annotate discovered fragments with gold at phone level.
    Change in place the (mutable) discovered fragments"""
    #TODO: faster -> sort by file, sort by onset, read only from prev onset
    phone_index = 0
    for fname in disc_dict:
        for disc_interval in disc_dict[fname]:
            transcription = []
            fname = disc_interval.fname
            transcription_started = False
            transcription_intervals = []
            # reading the corresponding gold transcription
            for phone_interval in gold_dict[fname][phone_index:]:
                if phone_interval.interval.overlaps_with(disc_interval.interval):
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
            frame_transcription = frame_alignement(
                transcription_intervals, disc_interval.interval)
            disc_interval.phone_transcription = transcription
            disc_interval.frame_transcription = frame_transcription


# def annotate_phone_disc(gold_dict, disc_dict):
#     """Annotate discovered fragments with gold at phone level"""
#     #TODO: faster -> sort by file, sort by onset, read only from prev onset
#     res = {}
#     for disc_class in disc_dict.keys():
#         res[disc_class] = []
#         for disc_interval in disc_dict[disc_class]:
#             transcription = []
#             fname = disc_interval.fname
#             transcription_started = False
#             transcription_intervals = []
#             # reading the corresponding gold transcription
#             for phone_interval in gold_dict[fname]:
#                 if phone_interval.interval.overlaps_with(disc_interval.interval):
#                     transcription_started = True
#                     transcription.append(phone_interval.label)
#                     transcription_intervals.append(phone_interval)
#                 elif transcription_started:
#                     # transcription started, but phone non overlap =>
#                     # end of transcription
#                     break
#             frame_transcription = frame_alignement(
#                 transcription_intervals, disc_interval.interval)
#             res[disc_class].append(TranscribedInterval(
#                 fname, disc_interval.interval, transcription,
#                 frame_transcription))
#     return res


def frame_alignement(interval_list, disc_interval):
    phones = [intervl.label for intervl in interval_list]
    onsets = np.array([intervl.interval.start for intervl in interval_list])
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
