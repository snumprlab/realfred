import glob
import pickle


def f(split=None):
    splits = ['valid_seen', 'valid_unseen', 'tests_seen', 'tests_unseen'] if split is None else [split]
    for split in splits:
        success, total = 0, 0
        succeess_task_type=[]
        for p in glob.glob(f'results/analyze_recs/{split}*'):
            r = pickle.load(open(p, 'rb'))
            success += sum([s['success'] for s in r])
            total += len(r)
        print(split, success, total, success / total)


for split in ['valid_seen', 'valid_unseen', 'tests_seen', 'tests_unseen']:
    print(f"---------- {split} --------------------")
    try:
        f( split)
    except:
        pass
