"""Compute several stats on discovered clusters"""

from __future__ import division
import tde.measures.nlp as nlp
import load
import itertools
import collections
import abnet.sampler as dtw
import numpy as np


def run(gold_file, disc_file):
    gold_phones = load.load_gold(gold_file)
    disc_frags = load.load_disc(disc_file)
    annotated_disc = load.annotate_phone_disc(gold_phones, disc_frags)
    compute_neds(annotated_disc)


IntervalPairs = collections.namedtuple('IntervalPairs',
                                       ['i1', 'i2', 'stats'])


def make_pairs(intervals_dict):
    """From a dictionnary of interval grouped by class, make all
    the possible (unordered) pairs of words
    """
    res = []
    for disc_class, disc_intervals in  intervals_dict.iteritems():
        for frag1, frag2 in itertools.combinations(disc_intervals, 2):
            res.append(IntervalPairs._make((
                frag1, frag2, {'class': disc_class})))
    return res


def compute_neds(annotated_pairs):
    """Compute phone level ned and frame level ned for each pair of
    word
    ned(x, y) = levenstein(x, y) / max(|x|, |y|)
    """
    for pair in annotated_pairs:
        pair.stats['phone_ned'] = nlp.ned(pair.i1.phone_transcription,
                                          pair.i2.phone_transcription)
        pair.stats['frame_ned'] = nlp.ned(pair.i1.frame_transcription,
                                          pair.i2.frame_transcription)


def compute_dtws(annotated_pairs, feature_file):
    """Compute dtw cosine distance between each pair of words
    """
    featuresAPI = dtw.FeaturesAPI(feature_file)
    for pair in annotated_pairs:
        segment1 = pair.i1.interval
        segment2 = pair.i2.interval
        dtws = featuresAPI.do_dtw((pair.i1.fname, segment1.start, segment1.end),
                                  (pair.i2.fname, segment2.start, segment2.end))
        pair.stats['dtw'] = dtws[0]
        pair.stats['temp_dist'] = temporal_distance(np.array(dtws[1]), np.array(dtws[2]))
        assert abs(len(pair.i1.frame_transcription) - dtws[1][-1] - 1) <= 1
        assert abs(len(pair.i2.frame_transcription) - dtws[2][-1] - 1) <= 1
        pair.stats['dtw_transcription'] = dtw_transcription_distance(
            pair.i1.frame_transcription, dtws[1],
            pair.i2.frame_transcription, dtws[2])


def temporal_distance2(path1, path2):
    """
    The temporal distance is defined as mean({1 if f'!=0, 0 otherwise})
    """
    x_diff = path1[1:] - path1[:-1]
    y_diff = path2[1:] - path2[:-1]
    indicator = np.logical_xor(x_diff, y_diff)
    return np.mean(indicator)


def temporal_distance(path1, path2, wlen=5):
    """Compute the "temporal" distance between 2 paths
    The temporal distance is defined as mean(abs(log(f')), f being
    the transformation so that f(path1) = path2

    wlen is the resolution (in number of frames) to calculate f' with
    (f is NOT approximated to piecewise linear on that resolution !!!
    only f' is approximated to the growth rate on that resolution)
    """
    assert len(path1) > 5
    x_diff = path1[wlen:] - path1[:wlen]
    y_diff = path2[wlen:] - path2[:wlen]
    f_prime = x_diff / y_diff
    return np.mean(np.abs(np.log(f_prime)))


def dtw_transcription_distance(transcription1, path1, transcription2, path2):
    """Compute the frame-wise distance of the dtw aligned transcriptions
    (dtw usually calculated with an other distance)
    """
    t1, t2 = np.array(transcription1), np.array(transcription2)
    assert path1[-1]+1 == t1.size or path1[-1]+2 == t1.size
    assert path2[-1]+1 == t2.size or path2[-1]+2 == t2.size, '{} {}'.format(path2[-1], t2.size)
    if path1[-1] + 1 != t1.size:
        t1 = t1[1:]
    if path2[-1] + 1 != t2.size:
        t2 = t2[1:]
    t1_align, t2_align = t1[path1], t2[path2]
    return np.mean(t1_align != t2_align)


def find_pos(pos_tagging_dict):
    raise NotImplementedError


def tests():
    gold_file = 'test/test.phn'
    disc_file = 'test/master_graph.zs'
    features_file = 'test/buckeye_fea_raw.h5f'
    gold_phones = load.load_gold(gold_file)
    disc_frags = load.load_disc(disc_file)
    disc_sorted_by_file = load.sort_disc(disc_frags)
    load.annotate_phone_disc(gold_phones, disc_sorted_by_file)
    pairs = make_pairs(disc_frags)
    compute_neds(pairs)
    compute_dtws(pairs, features_file)
    print pairs


if __name__ == '__main__':
    tests()
