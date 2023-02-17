import os
import re
import pathlib
from natsort import natsorted
from unidecode import unidecode
from datasets import load_dataset
from collections import defaultdict


CATEGORIES = ['Astronaut',
              'Airport',
              'Monument',
              'University',
              'Food',
              'SportsTeam',
              'City',
              'Building',
              'WrittenWork',
              'ComicsCharacter',
              'Politician',
              'Athlete',
              'MeanOfTransportation',
              'Artist',
              'CelestialBody']

NEW_CATEGORIES = ['MeanOfTransportation', 'CelestialBody', 'Politician', 'Athlete', 'Artist']

OLD_CATEGORIES = [item for item in CATEGORIES if item not in NEW_CATEGORIES]


def generate_files(data):
    # generate files per category
    for cat in CATEGORIES:
        
        data_reduced = data.filter(lambda x: x['category'] == cat)
        print(f'Reduced size for category {cat}:', len(data_reduced))

        # metric files generation; we use three references
        bleu_ref_files_gen(data_reduced, cat)
        meteor_3ref_files_gen(data_reduced, cat)
        ter_ref_files_gen(data_reduced, cat, True)

    # generate files per size
    for size in range(1, 8):
        
        data_reduced = data.filter(lambda x: x['size'] == size)
        print(f'Reduced set of triplet with size {size} is long:', len(data_reduced))

        bleu_ref_files_gen(data_reduced, str(size) + 'size')
        meteor_3ref_files_gen(data_reduced, str(size) + 'size')
        ter_ref_files_gen(data_reduced, str(size) + 'size', True)

    # generate files per type: old, new, all categories
    print('Gold count:', len(data))
    # metric files generation for all cats
    bleu_ref_files_gen(data, 'all-cat')
    meteor_3ref_files_gen(data, 'all-cat')
    ter_ref_files_gen(data, 'all-cat', True)

    data_reduced = data.filter(lambda x: x['category'] in NEW_CATEGORIES)
    print('Reduced (new) entry count:', len(data_reduced))
    # metric files generation for new cats
    bleu_ref_files_gen(data_reduced, 'new-cat')
    meteor_3ref_files_gen(data_reduced, 'new-cat')
    ter_ref_files_gen(data_reduced, 'new-cat', True)

    data_reduced = data.filter(lambda x: x['category'] in OLD_CATEGORIES)
    print('Reduced (old) entry count', len(data_reduced))
    # metric files generation for old cats
    bleu_ref_files_gen(data_reduced, 'old-cat')
    meteor_3ref_files_gen(data_reduced, 'old-cat')
    ter_ref_files_gen(data_reduced, 'old-cat', True)


def bleu_ref_files_gen(b_reduced, param):
    ids_refs = defaultdict(list)
    for entry in b_reduced:
        ids_refs[entry['eid']] = entry['lex']
    # length of the value with max elements
    max_refs = sorted(ids_refs.values(), key=len)[-1]
    # write references files for BLEU
    for j in range(0, len(max_refs)):
        with open('references/gold-' + param + '-reference' + str(j) + '.lex', 'w+') as f:
            out = ''
            # extract values sorted by key (natural sorting)
            values = [ids_refs[key] for key in natsorted(ids_refs.keys(), reverse=False)]
            for _, ref in enumerate(values):
                try:
                    # detokenise
                    lex_detokenised = ' '.join(re.split('(\W)', ref['text'][j]))
                    # delete redundant white spaces
                    lex_detokenised = ' '.join(lex_detokenised.split())
                    # convert to ascii and lowercase
                    out += unidecode(lex_detokenised.lower()) + '\n'
                except IndexError:
                    out += '\n'
                    lex_detokenised = ''
            f.write(out)

def meteor_3ref_files_gen(b_reduced, param):
    # data for meteor
    # For N references, it is assumed that the reference file will be N times the length of the test file,
    # containing sets of N references in order.
    # For example, if N=4, reference lines 1-4 will correspond to test line 1, 5-8 to line 2, etc.
    ids_refs = {}
    for entry in b_reduced:
        ids_refs[entry['eid']] = entry['lex']
    # maximum number of references
    max_refs = 3
    with open('references/gold-' + param + '-reference-3ref.meteor', 'w+') as f:
        # extract values sorted by key (natural sorting)
        values = [ids_refs[key] for key in natsorted(ids_refs.keys(), reverse=False)]
        for ref in values:
            empty_lines = max_refs - len(ref)  # calculate how many empty lines to add (e.g. 3 max references)
            out = ref['text'] # [lexicalis.lex for lexicalis in ref]
            out_clean = []
            for iter, sentence in enumerate(out):
                if iter < 3:
                    # detokenise
                    sent_clean = ' '.join(re.split('(\W)', sentence))
                    # delete redundant white spaces
                    sent_clean = ' '.join(sent_clean.split())
                    out_clean += [unidecode(sent_clean.lower())]
            f.write('\n'.join(out_clean) + '\n')
            if empty_lines > 0:
                f.write('\n' * empty_lines)


def ter_ref_files_gen(b_reduced, param, three_ref_only=False):
    # data for meteor
    # append (id1) to references
    out = ''
    for i, entry in enumerate(b_reduced):
        id_str = 'id' + str(i + 1)
        for i, lex in enumerate(entry['lex']['text']):
            # detokenise
            sent_clean = ' '.join(re.split('(\W)', lex))
            # delete redundant white spaces
            sent_clean = ' '.join(sent_clean.split())
            if three_ref_only and i > 2:        # three references maximum
                break
            out += unidecode(sent_clean.lower()) + ' (' + id_str + ')\n'
    if not three_ref_only:
        with open('references/gold-' + param + '-reference.ter', 'w+') as f:
            f.write(out)
    else:
        with open('references/gold-' + param + '-reference-3ref.ter', 'w+') as f:
            f.write(out)


def read_participant(data, output_file, teamname):
    # read participant's outputs
    output = []
    with open(output_file, 'r', encoding='utf-8') as f:
        output += [unidecode(line.strip()) for line in f if len(line.strip()) > 0]

    print(output[0], output[1])

    # per size
    for size in range(1, 8):
        entry_ids = []
        # look up id of a line in the gold benchmark, extract its size
        for entry in data:
            if int(entry['size']) == size:
                entry_ids += [int(entry['eid'][2:])]      # entry id -- 'Id1'
        output_reduced = [output[i-1] for i in sorted(entry_ids)]
        write_to_file(output_reduced, str(size)+'size', teamname)

    # per category
    for category in CATEGORIES:
        entry_ids = []
        # look up id of a line in the gold benchmark, extract its category
        for entry in data:
            if entry['category'] == category:
                entry_ids += [int(entry['eid'][2:])]      # entry id -- 'Id1'
        output_reduced = [output[i-1] for i in sorted(entry_ids)]
        write_to_file(output_reduced, category, teamname)

    # old categories
    entry_ids = []
    for category in OLD_CATEGORIES:
        # look up id of a line in the gold benchmark, extract its category
        for entry in data:
            if entry['category'] == category:
                entry_ids += [int(entry['eid'][2:])]      # entry id -- 'Id1'
    output_reduced = [output[i-1] for i in sorted(entry_ids)]
    write_to_file(output_reduced, 'old-cat', teamname)

    # new categories
    entry_ids = []
    for category in NEW_CATEGORIES:
        # look up id of a line in the gold benchmark, extract its category
        for entry in data:
            if entry['category'] == category:
                entry_ids += [int(entry['eid'][2:])]      # entry id -- 'Id1'
    output_reduced = [output[i-1] for i in sorted(entry_ids)]
    write_to_file(output_reduced, 'new-cat', teamname)
    
    # create all-category files
    write_to_file(output, 'all-cat', teamname)
    print('Files creating finished for: ', teamname)


def write_to_file(output_reduced, param, teamname):
    out = ''
    out_ter = ''
    for iter, item in enumerate(output_reduced):
        # detokenise, lowercase, and convert to ascii
        # we do this to ensure consistency between participants
        lex_detokenised = ' '.join(re.split('(\W)', unidecode(item.lower())))
        # delete redundant white spaces
        lex_detokenised = ' '.join(lex_detokenised.split())
        out += lex_detokenised + '\n'
        out_ter += lex_detokenised + ' (id' + str(iter + 1) + ')\n'

    with open('teams/' + teamname + '_' + str(param) + '.txt', 'w+') as f:
        f.write(out)
    with open('teams/' + teamname + '_' + str(param) + '_ter.txt', 'w+') as f:
        f.write(out_ter)


if __name__ == '__main__':
    all_data = load_dataset("web_nlg", 'webnlg_challenge_2017', split='test')
    all_data = all_data.filter(lambda x: x['test_category'] in ['testdata_with_lex', 'testdata_unseen_with_lex'])

    # generate references
    dir_name = 'references/'
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    if not os.listdir('references/'):
        generate_files(all_data)
    
    # read submissions and generate relative eval files
    if not os.path.exists('teams/') or not os.path.isdir('teams/'):
        os.makedirs('teams/')
    outfolder = 'submissions/'
    outfiles = list(pathlib.Path(outfolder).glob('*.txt'))
    teams = [outf.replace('.txt', '') for outf in outfiles]
    print('Producing evaluation files for the following teams: ', teams)
    with open('teams_list.txt', 'w') as of:
        of.write('\n'.join(teams))
    for outfile, team in zip(outfiles, teams):
        read_participant(all_data, outfile, team)

    folders_to_make = ['eval/', '../results/']
    for folder in folders_to_make:
        if not os.path.exists(folder) or not os.path.isdir(folder):
            os.makedirs(folder)
