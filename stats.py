"""Compute several stats on discovered clusters"""

from __future__ import division
from __future__ import print_function
import tde.measures.nlp as nlp
import load
import itertools
import collections
import numpy as np
import h5features
from dtw import DTW
import numba as nb

epsilon = 0.0001
SIL_LABEL = load.SIL_LABEL


def run(gold_file, disc_file):
    gold_phones = load.load_gold(gold_file)
    disc_frags = load.load_disc(disc_file)
    annotated_disc = load.annotate_phone_disc(gold_phones, disc_frags)
    compute_neds(annotated_disc)


IntervalPairs = collections.namedtuple('IntervalPairs',
                                       ['i1', 'i2', 'stats'])


class FeaturesAPI:
    """wrapper for h5features manipulation
    """
    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.index = h5features.read_index(feature_file)


    def get_features(self, segment):
        """return the features associated to a segment = (file, start, end)"""
        fileid, start, end = segment
        return h5features.read(self.feature_file, from_internal_file=fileid,
                                   from_time=start, to_time=end,
                                   index=self.index)[1][fileid]

    def get_features_from_file(self, fileid):
        """return the features accosiated to a file"""
        return h5features.read(self.feature_file, from_internal_file=fileid,
                               index=self.index)[1][fileid]
    
    def do_dtw(self, segment1, segment2):
        dtw = DTW(self.get_features(segment1), self.get_features(segment2),
                  return_alignment=1, python_dist_function=cosine_distance_numba)
        return dtw[0], dtw[-1][1], dtw[-1][2]

    def do_dtw_deciles(self, segment1, segment2, cut=10):
        features1, features2 = map(self.get_features, [segment1, segment2])
        dtw = DTW(self.get_features(segment1), self.get_features(segment2),
                  return_alignment=1, python_dist_function=cosine_distance_numba)
        dtw_dist_onpath = compute_distance_onpath(
            features1, features2, dtw[-1][1], dtw[-1][2], cosine_distance_numba)
        dtw_dist_cuts = np.array_split(dtw_dist_onpath, cut)
        assert (np.mean(dtw_dist_onpath) - dtw[0]/len(dtw[-1][1])) <= 0.001, '{}. {}'.format(np.mean(dtw_dist_onpath), dtw[0]/len(dtw[-1][1]))
        return np.mean(dtw_dist_onpath), dtw[-1][1], dtw[-1][2], [np.mean(x) for x in dtw_dist_cuts]


def compute_distance_onpath(feats1, feats2, path1, path2, dist_fun):
    assert len(path1) == len(path2)
    res = np.empty((len(path1),))
    for index, (i, j) in enumerate(zip(path1, path2)):
        res[index] = dist_fun(feats1[i, :], feats2[j, :])
    return res


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
        pair.stats['phone_ned'] = nlp.ned(
            pair.i1.phone_transcription,
            pair.i2.phone_transcription)
        pair.stats['frame_ned'] = nlp.ned(
            pair.i1.frame_transcription,
            pair.i2.frame_transcription)


def compute_silences(discovered_fragments):
    """Compute the proportion of silence in the discovered fragments
    """
    for fragments in discovered_fragments.itervalues():
        for fragment in fragments:
            fragment.stats['silence_proportion'] = fragment.frame_transcription.count('SIL') / len(fragment.frame_transcription)


def compute_dtws(annotated_pairs, feature_file, cut=10):
    """Compute dtw cosine distance between each pair of words
    """
    featuresAPI = FeaturesAPI(feature_file)
    for pair in annotated_pairs:
        segment1 = pair.i1.interval
        segment2 = pair.i2.interval
        dtws = featuresAPI.do_dtw_deciles(
            (pair.i1.fname, segment1.start, segment1.end),
            (pair.i2.fname, segment2.start, segment2.end))
        pair.stats['dtw'] = dtws[0]
        pair.stats['dtw_cuts'] = dtws[3]
        # dtw_cuts = np.array_split(dtws[0], cut)
        # pair.stats['dtw_deciles'] = [np.mean(x) for x in dtw_cuts]
        pair.stats['temp_dist'] = temporal_distance(np.array(dtws[1]), np.array(dtws[2]))
        pair.stats['norm_temp'], pair.stats['norm_temp_cuts'] = normalized_temporal_distance_cuts(np.array(dtws[1]), np.array(dtws[2]), cut=cut)
        # assert abs(len(pair.i1.frame_transcription) - dtws[1][-1] - 1) <= 1
        # assert abs(len(pair.i2.frame_transcription) - dtws[2][-1] - 1) <= 1
        (pair.stats['dtw_transcription'], pair.stats['frame_mismatch_deciles']) \
            = dtw_transcription_distance_cut(
                pair.i1.frame_transcription, dtws[1],
                pair.i2.frame_transcription, dtws[2],
                cut=cut)


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
    x_diff = path1[wlen:] - path1[:-wlen]
    y_diff = path2[wlen:] - path2[:-wlen] + epsilon
    f_prime = x_diff / y_diff
    return np.mean(np.abs(np.log(f_prime + epsilon)))


def normalized_temporal_distance(path1, path2, wlen=5):
    """Compute the "temporal" distance between 2 paths
    The temporal distance is defined as mean(abs(log(f')), f being
    the transformation so that f(path1) = path2

    wlen is the resolution (in number of frames) to calculate f' with
    (f is NOT approximated to piecewise linear on that resolution !!!
    only f' is approximated to the growth rate on that resolution)
    """
    assert len(path1) > 5
    x_diff = path1[wlen:] - path1[:-wlen]
    y_diff = path2[wlen:] - path2[:-wlen] + epsilon
    steep = path1[-1] / path2[-1]
    f_prime = x_diff / y_diff
    return np.mean(np.abs(np.log(f_prime / steep + epsilon)))


def normalized_temporal_distance_cuts(path1, path2, wlen=5, cut=10):
    """Compute the "temporal" distance between 2 paths
    The temporal distance is defined as mean(abs(log(f')), f being
    the transformation so that f(path1) = path2

    wlen is the resolution (in number of frames) to calculate f' with
    (f is NOT approximated to piecewise linear on that resolution !!!
    only f' is approximated to the growth rate on that resolution)
    """
    assert len(path1) > 5
    x_diff = path1[wlen:] - path1[:-wlen]
    y_diff = path2[wlen:] - path2[:-wlen] + epsilon
    steep = path1[-1] / path2[-1]
    f_prime = x_diff / y_diff
    temporal_distance = np.abs(np.log(f_prime / steep + epsilon))
    temporal_distance_cuts = np.array_split(temporal_distance, cut)
    return np.mean(temporal_distance), [np.mean(x) for x in temporal_distance_cuts]


def dtw_transcription_distance(transcription1, path1, transcription2, path2):
    """Compute the frame-wise distance of the dtw aligned transcriptions
    (dtw usually calculated with an other distance)
    """
    t1, t2 = np.array(transcription1), np.array(transcription2)
    # assert path1[-1]+1 == t1.size or path1[-1]+2 == t1.size
    # assert path2[-1]+1 == t2.size or path2[-1]+2 == t2.size, '{} {}'.format(path2[-1], t2.size)
    if path1[-1] + 2 == t1.size:
        t1 = t1[1:]
    if path2[-1] + 2 == t2.size:
        t2 = t2[1:]
    if path1[-1] == t1.size:
        t1 = np.append(t1, t1[-1])
    if path2[-1] == t2.size:
        t2 = np.append(t2, t2[-1])
    t1_align, t2_align = t1[path1], t2[path2]
    return np.mean(t1_align != t2_align)


def dtw_transcription_distance_cut(transcription1, path1, transcription2, path2, cut=10):
    """Compute the frame-wise distance of the dtw aligned transcriptions
    (dtw usually calculated with an other distance)
    return mean, mean-by-cut
    """
    t1, t2 = np.array(transcription1), np.array(transcription2)
    # assert path1[-1]+1 == t1.size or path1[-1]+2 == t1.size
    # assert path2[-1]+1 == t2.size or path2[-1]+2 == t2.size, '{} {}'.format(path2[-1], t2.size)
    if path1[-1] + 2 == t1.size:
        t1 = t1[1:]
    if path2[-1] + 2 == t2.size:
        t2 = t2[1:]
    if path1[-1] == t1.size:
        t1 = np.append(t1, t1[-1])
    if path2[-1] == t2.size:
        t2 = np.append(t2, t2[-1])
    t1_align, t2_align = t1[path1], t2[path2]
    assert t1_align.size == t2_align.size
    mismatch = t1_align != t2_align
    mismatch_cuts = np.array_split(mismatch, cut)
    return np.mean(mismatch), [np.mean(x) for x in mismatch_cuts]


def dtw_deciles(transcription1, path1, transcription2, path2, cut=10):
    raise NotImplementedError


def find_pos(pos_tagging_dict):
    raise NotImplementedError


@nb.autojit
def cosine_distance_numba(A, B):
    norm_A = 0
    norm_B = 0
    numerator = 0    
    for k in range(len(A)):
        numerator += A[k] * B[k]
        norm_A += A[k] * A[k]
        norm_B += B[k] * B[k]
    similarity = numerator / (np.sqrt(norm_A) * np.sqrt(norm_B))
    return (1 - similarity)/2

def cosine_distance_numpy(A, B):
    return 1 - (np.dot(A.T, B) / (np.linalg.norm(A) * np.linalg.norm(B)))


def tests():
    gold_file = 'test/new_english.phn'
    # disc_file = 'test/master_graph.zs'
    disc_file = 'test/english_disc'
    features_file = 'test/buckeye_fea_raw.h5f'
    gold_phones = load.load_gold(gold_file)
    disc_frags = load.load_disc(disc_file)
    disc_sorted_by_file = load.sort_disc(disc_frags)
    load.annotate_phone_disc(gold_phones, disc_sorted_by_file)
    compute_silences(disc_frags)
    pairs = make_pairs(disc_frags)
    compute_neds(pairs)
    compute_dtws(pairs, features_file)
    print_csv(pairs, 'res.txt')


def print_csv(annotated_pairs, csvfile):
    with open(csvfile, 'w') as fout:
        print('\t'.join([
            'file name i1', 'onset i1', 'offset i1',
            'file name i2', 'onset i2', 'offset i2',
            'transcription i1', 'transcription i2',
            'previous silence i1', 'next silence i1', 'proportion silence i1',
            'previous silence i2', 'next silence i2', 'proportion silence i2',
            'phone level ned', 'frame level ned', 'frame aligned mismatch',
            'temporal distance', 'normalized temporal distance',
            'normalized dtw cosine distance',
            'frame mismatch d1', 'frame mismatch d2',
            'frame mismatch d3', 'frame mismatch d4',
            'frame mismatch d5', 'frame mismatch d6',
            'frame mismatch d7', 'frame mismatch d8',
            'frame mismatch d9', 'frame mismatch d10',
            'cosine dist d1', 'cosine dist d2', 'cosine dist d3',
            'cosine dist d4', 'cosine dist d5', 'cosine dist d6',
            'cosine dist d7', 'cosine dist d8', 'cosine dist d9',
            'cosine dist d10',
            'norm temp dist d1', 'norm temp dist d2', 'norm temp dist d3',
            'norm temp dist d4', 'norm temp dist d5', 'norm temp dist d6',
            'norm temp dist d7', 'norm temp dist d8', 'norm temp dist d9',
            'norm temp dist d10',
        ]), file=fout)
        for pair in annotated_pairs:
            print('\t'.join(str(x) for x in [
                pair.i1.fname, pair.i1.interval.start, pair.i1.interval.end,
                pair.i2.fname, pair.i2.interval.start, pair.i2.interval.end,
                '-'.join(pair.i1.phone_transcription), '-'.join(pair.i2.phone_transcription),
                pair.i1.stats['prev_sil'], pair.i1.stats['next_sil'], pair.i1.stats['silence_proportion'],
                pair.i2.stats['prev_sil'], pair.i2.stats['next_sil'], pair.i2.stats['silence_proportion'],
                pair.stats['phone_ned'], pair.stats['frame_ned'],
                pair.stats['dtw_transcription'], pair.stats['temp_dist'],
                pair.stats['norm_temp'], pair.stats['dtw'],]
                            + pair.stats['frame_mismatch_deciles']
                            + pair.stats['dtw_cuts']
                            + pair.stats['norm_temp_cuts']
                        ),
                  file=fout)


if __name__ == '__main__':
    tests()
