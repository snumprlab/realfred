import os
import numpy as np

NUM_EPOCHS=50
score_table = -np.ones((NUM_EPOCHS,8))

for i in range(NUM_EPOCHS):
    for s in ['valid_seen', 'valid_unseen']:
        fname = 'log_{}_{}.txt'.format(i, s)
        if not os.path.exists(fname):
            continue
        with open(fname, 'r') as f:
            try:
                # scores: 0 (SR) 1 (PLW SR) 2 (GC) 3 (PLW GC)
                scores = f.readlines()[-5:-1]
                if any('-' in score for score in scores):
                    continue
                scores = [eval(score.split(' ')[-1]) for score in scores]

                dsplit = 0 if s == 'valid_seen' else 4 # unseen for else
                score_table[i][dsplit:dsplit+4] = np.array(scores)
            except:
                continue

score_table = score_table * 100. # to percentage

with open('result_table.txt', 'w') as f:
    f.write('\t\tSeen\t\t\t\tUnseen\n')
    f.write('epoch\tSR\t\tGC\t\tSR\t\tGC\n')
    for i in range(score_table.shape[0]):
        try:
            f.write('{}\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})\n'.format(
                i,
                '-' if score_table[i][0] < 0 else score_table[i][0],
                '-' if score_table[i][1] < 0 else score_table[i][1],
                '-' if score_table[i][2] < 0 else score_table[i][2],
                '-' if score_table[i][3] < 0 else score_table[i][3],
                '-' if score_table[i][4] < 0 else score_table[i][4],
                '-' if score_table[i][5] < 0 else score_table[i][5],
                '-' if score_table[i][6] < 0 else score_table[i][6],
                '-' if score_table[i][7] < 0 else score_table[i][7],
            ))
        except:
            continue

