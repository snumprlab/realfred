import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, required=True)
parser.add_argument('--json_name', type=str)

args = parser.parse_args()

if args.json_name is None:
	args.json_name = args.dn_startswith

results = {'tests_seen': [], 'tests_unseen': []}
for seen_str in ['seen', 'unseen']:
	pickle_globs = glob("results/leaderboard/actseqs_test_" + seen_str + "_" + args.dn_startswith + "*")
	print(pickle_globs)
	pickles = []
	analysis_recs = []
	for g in pickle_globs:
		f, t = g.split('/')[-1][:-2].split('_')[-2:]
		analysis_rec = pickle.load(open('results/analyze_recs/tests_{}_anaylsis_recs_from_{}_to_{}_{}.p'.format(seen_str, f, t, args.dn_startswith), 'rb'))
		pickles += pickle.load(open(g, 'rb'))
		analysis_recs += analysis_rec
	
	#
	assert len(pickles) == len(analysis_recs), 'not same'
	for i in range(len(analysis_recs)):
		#if analysis_recs[i]['task_type'] == 'pick_two_obj_and_place':
		#	continue
		if analysis_recs[i]['success']:
			for k in pickles[i]:
				n, trial_i = k
			for j in range(i + 1, len(analysis_recs)):
				for k in pickles[j]:
					n, trial_j = k
				if trial_i == trial_j and not analysis_recs[j]['success']:
					pickles[j] = pickles[i]

	total_logs =[]
	for i, t in enumerate(pickles):
		key = list(t.keys())[0]
		actions = t[key]
		trial = key[1]
		total_logs.append({trial:actions})

	for i, t in enumerate(total_logs):
		key = list(t.keys())[0]
		actions = t[key]
		new_actions = []
		for action in actions:
			if action['action'] == 'LookDown_0' or action['action'] == 'LookUp_0':
				pass
			else:
				new_actions.append(action)
		total_logs[i] = {key: new_actions}

	results['tests_'+seen_str] = total_logs
	

#assert len(results['tests_seen']) == 1533, f"The current tests_seen is {len(results['tests_seen'])}"
#assert len(results['tests_unseen']) == 1529, f"The current tests_seen is {len(results['tests_unseen'])}"

if not os.path.exists('leaderboard_jsons_oneForAll'):
	os.makedirs('leaderboard_jsons_oneForAll')

save_path = 'leaderboard_jsons_oneForAll/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
	json.dump(results, r, indent=4, sort_keys=True)

