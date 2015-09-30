import load
import stats

if __name__ == '__main__':
    gold_file = 'test/new_english.phn'
    # disc_file = 'test/master_graph.zs'
    # disc_file = 'test/english_disc'
    disc_file = 'test/500ms_gold.classes'
    # disc_file = 'test/plp_fea/pairs_l50_vad_2.txt'
    features_file = 'test/buckeye_fea_raw.h5f'
    gold_phones = load.load_gold(gold_file)
    disc_frags = load.load_disc(disc_file)
    # disc_frags = load.load_disc_pairs(disc_file)
    disc_sorted_by_file = load.sort_disc(disc_frags)
    load.annotate_phone_disc(gold_phones, disc_sorted_by_file)
    stats.compute_silences(disc_frags)
    pairs = stats.make_pairs(disc_frags)
    stats.compute_neds(pairs)
    stats.compute_dtws(pairs, features_file)
    stats.print_csv(pairs, 'res_500ms_gold.txt')
