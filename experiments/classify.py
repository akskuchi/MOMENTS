import argparse
import json
import os
import random
import time
from datetime import datetime

import models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

batch_sizes = {'L': 40, 'A': 8, 'V': 1, 'AL': 5, 'AV': 1, 'LV': 1, 'ALV': 1}


def valid_modalities(model_type, modalities):
	if model_type == 'OmniM':
		return modalities in batch_sizes.keys()
	if model_type == 'LM':
		return 'V' or 'A' not in modalities
	if model_type == 'ALM':
		return 'V' not in modalities
	if model_type == 'VLM':
		return 'A' not in modalities
	if model_type == 'Baseline':
		return modalities in ['A', 'L', 'V']


def classify(classifier, modalities, data, results_file, batch_size=1):
	results = {}
	for m in modalities:
		batch_size = batch_sizes[m]
		print(f'using modalities: {m} (batch size: {batch_size})')
		model_outputs = {}
		batch_ids, batch = [], []
		for moment_id, [video_path, audio_path, transcription_path] in tqdm(data.items()):
			transcription = json.load(open(transcription_path))['global'].strip()
			if transcription in [
				'',
				'. . . . . . .',
				'. . .',
				'. . . . . . . . . . . .',
				'. . . . .',
				'. . . . . .',
				'. . . . . . . .',
				'. . . . . . . . .',
				'. . . . . . . . . .',
				'. . . .',
				'. . . . . . . . . . . . . . .',
			]:
				transcription = json.load(open(transcription_path))['local'].strip()
			batch_ids.append(moment_id)
			batch.append({
				'A': None if 'A' not in m else audio_path,
				'L': None if 'L' not in m else transcription,
				'V': None if 'V' not in m else video_path,
			})
			if len(batch) == batch_size:
				processed_responses, probabilities, raw_responses = classifier.get_model_output(
					classifier.construct_input(batch)
				)
				for idx, moment_id in enumerate(batch_ids):
					model_outputs[moment_id] = {
						'gt_label': 0 if 'NIM' in moment_id else 1,
						'gen_label': processed_responses[idx],
						'probabilities': probabilities[idx],
						'gen_response': raw_responses[idx],
					}
				results[m] = {'model_outputs': model_outputs}
				with open(results_file, 'w') as fh:
					json.dump(results, fh, indent=4)
				fh.close()
				batch_ids.clear(), batch.clear()
		if len(batch_ids) != 0:
			processed_responses, probabilities, raw_responses = classifier.get_model_output(
				classifier.construct_input(batch)
			)
			for idx, moment_id in enumerate(batch_ids):
				model_outputs[moment_id] = {
					'gt_label': 0 if 'NIM' in moment_id else 1,
					'gen_label': processed_responses[idx],
					'probabilities': probabilities[idx],
					'gen_response': raw_responses[idx],
				}
			batch_ids.clear(), batch.clear()
		results[m] = {
			'model_outputs': model_outputs,
		}
		print(
			f'\tF1-score: {results[m]["evaluation_results"]["F1-score"]:.3f}, accuracy: {results[m]["evaluation_results"]["accuracy"]:.3f}'
		)
		with open(results_file, 'w') as fh:
			json.dump(results, fh, indent=4)
		fh.close()


if __name__ == '__main__':
	print('\n====[classification of MOMENTS]====\n')
	start_time = time.time()
	date = datetime.today().strftime('%Y-%m-%d')

	parser = argparse.ArgumentParser()
	parser.add_argument('--moments_path', default='data', help='place the dataset (game ID dirs) under this path')
	parser.add_argument(
		'--subset',
		default=None,
		type=int,
		help='number of game ids to consider (<=100)',
	)
	parser.add_argument('--game_id', default=None, type=str, help='overrides --subset')
	parser.add_argument('--model_type', default='LM', choices=['OmniM', 'ALM', 'LM', 'VLM', 'Baseline', 'BaselineInference'])
	parser.add_argument('--modalities', default='L', help='example for OmniM: A|L|V|AL|AV|LV|ALV')
	parser.add_argument(
		'--model_name',
		default='meta-llama/Meta-Llama-3.1-8B-Instruct',
		choices=[
			'Qwen/Qwen2.5-Omni-7B',
			'Qwen/Qwen2.5-Omni-3B',
			'Qwen/Qwen3-Omni-30B-A3B-Instruct',
			'Qwen/Qwen2.5-7B-Instruct',
			'Qwen/Qwen3-4B-Instruct-2507',
			'meta-llama/Meta-Llama-3.1-8B-Instruct',
			'Qwen/Qwen2.5-VL-7B-Instruct',
			'Qwen/Qwen3-VL-4B-Instruct',
			'Qwen/Qwen2-Audio-7B-Instruct',
			'mistralai/Voxtral-Mini-3B-2507',
		],
	)
	parser.add_argument('--system_prompt_id', default='s2', choices=['s1', 's2'])
	parser.add_argument('--generation_prompt_id', default='g1', choices=['g1', 'g2'])
	parser.add_argument(
		'--results_path', default='results', help='path where model responses, checkpoints, and analyses plots are located'
	)
	args = parser.parse_args()

	moments_dataset = json.load(open('data.json'))['moments_dataset']
	version = 'v2'
	model_class = getattr(models, args.model_type if 'Inference' not in args.model_type else 'Baseline')
	classifier = model_class(args)

	if 'Baseline' in args.model_type:
		if os.path.exists(f'{args.results_path}/Baseline/data_splits.json'):
			baseline_data_splits = json.load(open(f'{args.results_path}/Baseline/data_splits.json'))
			train_ids, test_ids = (
				baseline_data_splits['train_ids'],
				baseline_data_splits['test_ids'],
			)
		else:
			train_ids, test_ids = train_test_split(list(moments_dataset.keys()), test_size=0.25, random_state=42)
			with open(f'{args.results_path}/Baseline/data_splits.json', 'w') as fh:
				json.dump({'train_ids': train_ids, 'test_ids': test_ids}, fh, indent=4)
			fh.close()

		X_train, X_test, y_train, y_test = [], [], [], []
		for k in train_ids:
			gid, half, moment_name = k.split('-')
			category = 'non-important-moments' if moment_name.startswith('NIM_') else 'important-moments'
			X_train.append(
				f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}_{version}.json'
				if args.modalities == 'L'
				else f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}_{version}.wav'
				if args.modalities == 'A'
				else f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}.mp4'
			)
			y_train.append(moments_dataset[k])
		for k in test_ids:
			gid, half, moment_name = k.split('-')
			category = 'non-important-moments' if moment_name.startswith('NIM_') else 'important-moments'
			X_test.append(
				f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}_{version}.json'
				if args.modalities == 'L'
				else f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}_{version}.wav'
				if args.modalities == 'A'
				else f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}.mp4'
			)
			y_test.append(moments_dataset[k])

		if 'Inference' in args.model_type:
			y_probs = classifier.get_model_responses(X_test, f'{args.results_path}/Baseline/{args.modalities}_{version}')
			responses = {test_id: y_probs[idx] for idx, test_id in enumerate(test_ids)}
			save_to = f'{args.results_path}/Baseline/{args.modalities}_{version}_responses.json'
			with open(save_to, 'w') as fh:
				json.dump(responses, fh, indent=4)
			fh.close()
			print(f'saved responses to: {save_to}')
		else:
			classifier.train_test_eval(
				X_train,
				X_test,
				y_train,
				y_test,
				f'{args.results_path}/Baseline/{args.modalities}_{version}',
			)
	else:
		game_ids = os.listdir(args.moments_path)
		if args.subset is not None:
			game_ids = random.sample(game_ids, args.subset)
		if args.game_id is not None:
			game_ids = [args.game_id]
		print(f'considering {len(game_ids)} game{"s" if len(game_ids) > 1 else ""} from the dataset.\n')

		assert valid_modalities(args.model_type, args.modalities), (
			f'[ERROR] model type ({args.model_type}) and modalities ({args.modalities}) mismatch!'
		)
		modalities = args.modalities.split('|')
		data = {}
		for k in moments_dataset.keys():
			gid, half, moment_name = k.split('-')
			if gid not in game_ids:
				continue
			category = 'non-important-moments' if moment_name.startswith('NIM_') else 'important-moments'
			data[k] = [
				f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}.mp4',
				f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}_{version}.wav',
				f'{args.moments_path}/{gid}/{category}/{half}/{moment_name}_{version}.json',
			]
		if len(game_ids) == 100:
			assert len(data) == 3954, '[ERROR] something wrong with data!'
		results_file = f'{args.results_path}/{args.model_type}/{version}_{args.modalities.replace("|", "_")}_{args.system_prompt_id}_{args.generation_prompt_id}_{args.model_name.split("/")[1]}.json'
		classify(classifier, modalities, data, results_file)
		print(f'\nsaved results to: {results_file}')

	end_time = time.time()
	running_time = end_time - start_time
	minutes = int(running_time // 60)
	seconds = running_time % 60
	print(f'\n==> running time: {minutes} minutes and {seconds:.2f} seconds')
