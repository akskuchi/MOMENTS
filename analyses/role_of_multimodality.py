import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import get_logit_differences, models


def get_confidences_df(args, moment_ids, category):
	confidences = {}
	combs, _ = models[args.model_name]
	for comb in combs:
		logit_differences = get_logit_differences(args, comb)
		for k in moment_ids:
			if k not in confidences:
				confidences[k] = {'category': category}
			confidences[k][comb] = logit_differences[k]
	return pd.DataFrame.from_dict(confidences)


def plot_unimodal_vs_multimodal(df, save_to=None):
	plot_df = df.T.copy()

	score_cols = list(df.index)[1:]
	plot_df[score_cols] = plot_df[score_cols].apply(pd.to_numeric)

	unimodal_settings, multimodal_settings = [], []
	for col in score_cols:
		if len(col) == 1:
			unimodal_settings.append(col)
		else:
			multimodal_settings.append(col)
	plot_df['max_unimodal'] = plot_df[unimodal_settings].max(axis=1)
	plot_df['max_multimodal'] = plot_df[multimodal_settings].max(axis=1)

	palette_map = {
		'GOAL': 'xkcd:cerulean blue',
		'SHOT-ON-TARGET': 'xkcd:pumpkin orange',
		'CORNER/THROW-IN': 'xkcd:pumpkin orange',
	}
	markers_map = {'GOAL': '8', 'SHOT-ON-TARGET': 'X', 'CORNER/THROW-IN': 's'}

	plt.figure(figsize=(8, 6), dpi=100, constrained_layout=True)
	sns.set_style('whitegrid')
	sns.scatterplot(
		data=plot_df,
		x='max_unimodal',
		y='max_multimodal',
		hue='category',
		style='category',
		palette=palette_map,
		markers=markers_map,
		s=100,
		alpha=0.8,
		edgecolor='k',
	)

	min_val, max_val = (
		plot_df[['max_unimodal', 'max_multimodal']].min().min(),
		plot_df[['max_unimodal', 'max_multimodal']].max().max(),
	)
	plt.plot([min_val, max_val], [min_val, max_val], ls='--', c='.3')

	plt.xlabel('max unimodal confidence', fontsize=16)
	plt.ylabel('max multimodal confidence', fontsize=16)
	plt.legend(loc='upper left', ncol=1, fontsize=16)
	if save_to is not None:
		plt.savefig(save_to)
		print(f'==> saved multimodal vs. unimodal plot to: {save_to}\n')
	plt.show()


if __name__ == '__main__':
	print('\n====[Role of Multimodality]====\n')
	start_time = time.time()
	date = datetime.today().strftime('%Y-%m-%d')

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model_name',
		default='Qwen2.5-Omni-7B',
		choices=['Qwen2.5-Omni-7B', 'Qwen2.5-VL-7B-Instruct'],
		help='pass a model name from this list (if the classification results are already available)',
	)
	parser.add_argument(
		'--results_path', default='results', help='path where model responses, checkpoints, and analyses plots are located'
	)
	args = parser.parse_args()

	category_annotations = json.load(open(Path(__file__).parent / 'category_annotations.json'))
	confidences_df = pd.concat(
		[
			get_confidences_df(args, list(category_annotations['GOAL']['IMs']), 'GOAL'),
			get_confidences_df(
				args,
				list(category_annotations['SHOT-ON-TARGET']['IMs']) + list(category_annotations['SHOT-ON-TARGET']['NIMs']),
				'SHOT-ON-TARGET',
			),
			get_confidences_df(
				args,
				list(category_annotations['CORNER/THROW-IN']['IMs']) + list(category_annotations['CORNER/THROW-IN']['NIMs']),
				'CORNER/THROW-IN',
			),
		],
		axis=1,
	)

	plot_unimodal_vs_multimodal(
		confidences_df,
		f'{args.results_path}/role_of_multimodality_{args.model_name}.pdf',
	)

	end_time = time.time()
	running_time = end_time - start_time
	minutes = int(running_time // 60)
	seconds = running_time % 60
	print(f'\n==> running time: {minutes} minutes and {seconds:.2f} seconds')
