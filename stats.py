"""Compute several stats on discovered clusters"""

import tde.measures.nlp as nlp
import load
import itertools
import collections
import abnet.sampler as dtw


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


def temporal_distance(path1, path2):
    """Compute the "temporal" distance between 2 paths
    The temporal distance is defined as mean(abs(log(f')), f being
    the transformation so that f(path1) = path2
    """


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
