import argparse
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.utils import resample

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyses.utils import load_model_responses, models


def interpret_baseline_text(text=None):
	if text is None:
		text = "I think it was a cross, because he's run around it, but he's whipped it in anyway. I think Leno had a cover flight, didn't he?"

	model, vect = (
		joblib.load('results/Baseline/L_v2_model.joblib'),
		joblib.load('results/Baseline/L_v2_vectorizer.joblib'),
	)
	X_sample = vect.transform([text])
	feature_names = vect.get_feature_names_out()

	weights = {feature_names[i]: model.coef_[0][i] for i in X_sample.indices}
	return sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)


def get_f1(y_true, y_probs):
	return f1_score(y_true, [1 if probs[1] > probs[0] else 0 for probs in y_probs])


def get_cf_matrix(y_true, y_probs):
	return confusion_matrix(y_true, [1 if probs[1] > probs[0] else 0 for probs in y_probs])


def get_mcc(y_true, y_probs):
	return matthews_corrcoef(y_true, [1 if probs[1] > probs[0] else 0 for probs in y_probs])


def get_accuracy(y_true, y_probs):
	return accuracy_score(y_true, [1 if probs[1] > probs[0] else 0 for probs in y_probs])


def get_roc_auc(y_true, y_probs):
	y_probs = [prob[1] / 2 for prob in y_probs]
	return roc_auc_score(y_true, y_probs)


def get_f1_with_ci(y_true, y_probs, n_boots=1000):
	original_score = get_f1(y_true, y_probs)
	scores = []
	for _ in range(n_boots):
		indices = resample(np.arange(len(y_true)), n_samples=len(y_true), replace=True)
		scores.append(get_f1(np.asarray(y_true)[indices], np.asarray(y_probs)[indices]))
	lower, upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
	margin = (upper - lower) / 2
	return original_score, margin.item()


def get_accuracy_with_ci(y_true, y_probs, n_boots=1000):
	original_score = get_accuracy(y_true, y_probs)
	scores = []
	for _ in range(n_boots):
		indices = resample(np.arange(len(y_true)), n_samples=len(y_true), replace=True)
		scores.append(get_accuracy(np.asarray(y_true)[indices], np.asarray(y_probs)[indices]))
	lower, upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
	margin = (upper - lower) / 2
	return original_score, margin.item()


def get_mcc_with_ci(y_true, y_probs, n_boots=1000):
	original_score = get_mcc(y_true, y_probs)
	scores = []
	for _ in range(n_boots):
		indices = resample(np.arange(len(y_true)), n_samples=len(y_true), replace=True)
		scores.append(get_mcc(np.asarray(y_true)[indices], np.asarray(y_probs)[indices]))
	lower, upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
	margin = (upper - lower) / 2
	return original_score, margin.item()


def get_roc_with_ci(y_true, y_probs, n_boots=1000):
	original_score = get_roc_auc(y_true, y_probs)
	scores = []
	for _ in range(n_boots):
		indices = resample(np.arange(len(y_true)), n_samples=len(y_true), replace=True)
		scores.append(get_roc_auc(np.asarray(y_true)[indices], np.asarray(y_probs)[indices]))
	lower, upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
	margin = (upper - lower) / 2
	return original_score, margin.item()


if __name__ == '__main__':
	print('\n====[Evaluating Classifier Responses]====\n')
	start_time = time.time()
	date = datetime.today().strftime('%Y-%m-%d')

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model_name',
		default='Qwen2.5-Omni-7B',
		choices=list(models.keys()),
		help='pass a model name from this list (if the classification results are already available)',
	)
	parser.add_argument('--results_path', default='results')
	args = parser.parse_args()

	print(f'model: {args.model_name}')
	moment_ids = json.load(open('data.json'))['moments_dataset'].keys()
	combs, model_type = models[args.model_name]
	for m_comb in combs:
		print(f'\t{m_comb}', end=' = ')
		responses = load_model_responses(model_type, args.model_name, m_comb, args.results_path)
		y_true = (
			[responses['g1'][k]['gt_label'] for k in moment_ids]
			if model_type != 'Baseline'
			else ['NIM_' not in k for k, _ in responses.items()]
		)
		y_probs = (
			[
				list(map(sum, zip(responses['g1'][k]['probabilities'], responses['g2'][k]['probabilities'])))
				for k in moment_ids
			]
			if model_type != 'Baseline'
			else list(responses.values())
		)
		print(f'MCC: {get_mcc(y_true, y_probs)}, accuracy: {get_accuracy(y_true, y_probs)}')

	end_time = time.time()
	running_time = end_time - start_time
	minutes = int(running_time // 60)
	seconds = running_time % 60
	print(f'\n==> running time: {minutes} minutes and {seconds:.2f} seconds')
