import json
from statistics import mean

import numpy as np

models = {
	'Qwen2.5-Omni-7B': (['A', 'L', 'V', 'AL', 'AV', 'LV', 'ALV'], 'OmniM'),
	'Qwen3-Omni-30B-A3B-Instruct': (['A', 'L', 'V', 'AL', 'AV', 'LV', 'ALV'], 'OmniM'),
	'baseline_video': (['V'], 'Baseline'),
	'Qwen2.5-VL-7B-Instruct': (['V', 'L', 'LV'], 'VLM'),
	'Qwen3-VL-4B-Instruct': (['V', 'L', 'LV'], 'VLM'),
	'baseline_audio': (['A'], 'Baseline'),
	'Qwen2-Audio-7B-Instruct': (['A', 'L', 'AL'], 'ALM'),
	'Voxtral-Mini-3B-2507': (['A', 'L', 'AL'], 'ALM'),
	'baseline_text': (['L'], 'Baseline'),
	'Qwen2.5-7B-Instruct': (['L'], 'LM'),
	'Qwen3-4B-Instruct-2507': (['L'], 'LM'),
	'Meta-Llama-3.1-8B-Instruct': (['L'], 'LM'),
}

def load_model_responses(
	model_type='OmniM',
	model_name='Qwen2.5-Omni-7B',
	modalities='L',
	results_path='results',
):
	if model_type == 'Baseline':
		return json.load(open(f'{results_path}/{model_type}/{modalities}_v2_responses.json'))
	return {
		p: json.load(open(f'{results_path}/{model_type}/v2_{modalities}_s2_{p}_{model_name}.json'))[modalities][
			'model_outputs'
		]
		for p in ['g1', 'g2']
	}


def get_logit_differences(args, comb=None):
	def get_logit_difference(label, probs_g1, probs_g2):
		return mean([
			np.log(probs_g1[label] / probs_g1[1 - label]).item(),
			np.log(probs_g2[label] / probs_g2[1 - label]).item(),
		])

	combs, model_type = models[args.model_name]
	if comb is None:
		comb = combs[0]
	responses = load_model_responses(model_type, args.model_name, comb, args.results_path)
	if model_type == 'Baseline':
		return {
			k: np.log(probs[0 if 'NIM_' in k else 1] / probs[1 - (0 if 'NIM_' in k else 1)]).item()
			for k, probs in responses.items()
		}
	return {
		k: get_logit_difference(
			responses['g1'][k]['gt_label'],
			responses['g1'][k]['probabilities'],
			responses['g2'][k]['probabilities'],
		)
		for k in responses['g1'].keys()
	}