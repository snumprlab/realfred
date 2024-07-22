import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, default="debug")
parser.add_argument('--json_name',default='Template+ALLHeuristic+noappended' ,type=str)

args = parser.parse_args()

if args.json_name is None:
	args.json_name = args.dn_startswith

results = {'tests_seen': [], 'tests_unseen': []}
for seen_str in ['seen', 'unseen']:
	pickle_globs = glob("results/leaderboard/actseqs_test_" + seen_str + "_" + args.dn_startswith + "*")
	print(pickle_globs)
	pickles = []
	for g in pickle_globs:
		print(g +"       :       " +str(len(pickle.load(open(g, 'rb')))))		
		pickles += pickle.load(open(g, 'rb'))

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
print('seen      :    '  +str(len(results['tests_seen']))+'/1533')
print('unseen      :    '  +str(len(results['tests_unseen']))+'/1529')	

assert len(results['tests_seen']) == 1533, f"The current tests_seen is {len(results['tests_seen'])}"
assert len(results['tests_unseen']) == 1529, f"The current tests_seen is {len(results['tests_unseen'])}"

if not os.path.exists('leaderboard_jsons'):
	os.makedirs('leaderboard_jsons')

save_path = 'leaderboard_jsons/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
	json.dump(results, r, indent=4, sort_keys=True)

