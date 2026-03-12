import argparse
import time
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import get_logit_differences, models


def get_color(name, ratio):
	rgb = mcolors.to_rgb(f'xkcd:{name}')
	return mcolors.rgb2hex([(1.0 - ratio) * c + ratio * 1.0 for c in rgb])


def get_modality_contribution_scores(args):
	combs, model_type = models[args.model_name]
	modalities_in_scope = 3 if model_type == 'OmniM' else 2 if model_type in ['ALM', 'VLM'] else 1
	sample_level_contribution_scores = {}
	logit_differences = {comb: get_logit_differences(args, comb) for comb in combs}
	moment_ids = logit_differences[combs[0]].keys()
	for k in moment_ids:
		sample_level_contribution_scores[k] = {
			m: (
				sum([logit_differences[comb][k] for comb in combs if m in comb])
				- sum([logit_differences[comb][k] for comb in combs if m not in comb])
			)
			/ sum([1 if m in comb else 0 for comb in combs])
			for m in combs[:modalities_in_scope]
		}
	return sample_level_contribution_scores


def plot_contribution_scores(scores, save_to=None):
	moment_ids = list(scores.keys())
	modalities_in_scope, modality_names = (
		scores[moment_ids[0]].keys(),
		{'A': 'audio', 'V': 'video', 'L': 'text'},
	)
	data = {
		f'scores_{m}{m_type}': [scores[k][m] for k in moment_ids if m_type in k]
		for m in modalities_in_scope
		for m_type in ['-IM', '-NIM']
	}
	df_A = pd.DataFrame({m: data[f'scores_{m}-IM'] for m in modalities_in_scope} | {'class': 'IM'})
	df_B = pd.DataFrame({m: data[f'scores_{m}-NIM'] for m in modalities_in_scope} | {'class': 'NIM'})
	df = pd.concat([df_A, df_B], ignore_index=True)
	df_melt = df.melt(
		id_vars=['class'],
		value_vars=modalities_in_scope,
		var_name='modality',
		value_name='score',
	)
	df_melt['group'] = df_melt['modality'] + '_' + df_melt['class']
	colors, strength = (
		{'L': 'cerulean blue', 'A': 'pumpkin orange', 'V': 'shamrock green'},
		{'IM': 0.35, 'NIM': 0.75},
	)
	custom_palette = {
		f'{m}_{m_type}': get_color(colors[m], strength[m_type]) for m in modalities_in_scope for m_type in ['IM', 'NIM']
	}
	class_order = [f'{m}_{m_type}' for m_type in ['IM', 'NIM'] for m in modalities_in_scope]

	plt.figure(figsize=(8, 4), constrained_layout=True, dpi=100)
	sns.violinplot(
		data=df_melt,
		x='group',
		y='score',
		hue='group',
		palette=custom_palette,
		order=class_order,
		inner='quartile',
		linewidth=1,
		density_norm='width',
		legend=False,
	)
	sns.pointplot(
		data=df_melt,
		x='group',
		y='score',
		hue='group',
		palette=['xkcd:black'] * len(custom_palette),
		order=class_order,
		errorbar=None,
		linestyle='none',
		marker='D',
		markersize=8,
		legend=False,
	)
	plt.gca().set_xticks(range(len(class_order)))
	plt.gca().set_xticklabels(
		[modality_names[c.split('_')[0]] + f'\n({c.split("_")[1]})' for c in class_order],
		rotation=0,
		ha='center',
		fontsize=16,
	)
	ax = plt.gca()
	for spine in ax.spines.values():
		spine.set_visible(False)
	separator_at = 2.5 if len(modalities_in_scope) == 3 else 1.5 if len(modalities_in_scope) == 2 else 0.5
	plt.axvline(x=separator_at, color='xkcd:silver', linestyle='-', linewidth=1, alpha=0.7)
	plt.axhline(0, color='red', linestyle='-', linewidth=0.8, alpha=0.7, label='no contribution')
	legend_handles = [
		plt.Line2D(
			[0],
			[0],
			color='black',
			marker=None,
			linestyle='--',
			markersize=10,
		),
		plt.Line2D([0], [0], color='black', marker='D', linestyle='None', markersize=8),
	]
	legend_labels = ['median', 'mean']
	plt.legend(
		handles=legend_handles,
		labels=legend_labels,
		loc='upper center',
		ncol=2,
		bbox_to_anchor=(0.5, 1.05),
		fontsize=16,
		columnspacing=0.8,
		frameon=True,
		shadow=True,
	)
	plt.ylabel('')
	plt.xlabel('')
	plt.grid(axis='y', alpha=0.25)
	if save_to is not None:
		plt.savefig(save_to)
		print(f'==> saved modality-level contribution scores to: {save_to}\n')
	plt.show()


if __name__ == '__main__':
	print('\n====[Influence of Modalities]====\n')
	start_time = time.time()
	date = datetime.today().strftime('%Y-%m-%d')

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model_name',
		default='Qwen2.5-Omni-7B',
		choices=list(models.keys()),
		help='pass a model name from this list (if the classification results are already available)',
	)
	parser.add_argument(
		'--results_path', default='results', help='path where model responses, checkpoints, and analyses plots are located'
	)
	args = parser.parse_args()

	plot_contribution_scores(
		get_modality_contribution_scores(args),
		f'{args.results_path}/influence_of_modalities_{args.model_name}.pdf',
	)

	end_time = time.time()
	running_time = end_time - start_time
	minutes = int(running_time // 60)
	seconds = running_time % 60
	print(f'\n==> running time: {minutes} minutes and {seconds:.2f} seconds')
