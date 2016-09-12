from __future__ import print_function
import load
import stats
import sys


usage = 'python run_ned.py gold_annotations discovered_pairs output_file'

def print_tsv(annotated_pairs, tsvfile):
    with open(tsvfile, 'w') as fout:
        print('\t'.join([
            'file name i1', 'onset i1', 'offset i1',
            'file name i2', 'onset i2', 'offset i2',
            'transcription i1', 'transcription i2',
            'phone level ned']))
        for pair in annotated_pairs:
            print('\t'.join(str(x) for x in [
                pair.i1.fname, pair.i1.interval.start, pair.i1.interval.end,
                pair.i2.fname, pair.i2.interval.start, pair.i2.interval.end,
                '-'.join(pair.i1.phone_transcription),
                '-'.join(pair.i2.phone_transcription),
                pair.stats['phone_ned']]),
                  file=fout)


if __name__ == '__main__':
    assert len(sys.argv) == 4, usage
    gold_file = sys.argv[1]
    disc_file = sys.argv[2]
    gold_phones = load.load_gold(gold_file)
    # disc_frags = load.load_disc(disc_file)
    disc_frags = load.load_disc_pairs(disc_file)
    disc_sorted_by_file = load.sort_disc(disc_frags)
    load.annotate_phone_disc(gold_phones, disc_sorted_by_file)
    # stats.compute_silences(disc_frags)
    pairs = stats.make_pairs(disc_frags)
    stats.compute_neds(pairs)
    print_tsv(pairs, sys.argv[3])
