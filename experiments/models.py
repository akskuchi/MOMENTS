import json
import logging
import os
import warnings
from io import BytesIO

import cv2
import joblib
import librosa
import numpy as np
import torch
from nvitop import select_devices
from PIL import Image
from prompts import generation_prompts, system_prompts
from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import (
	AutoImageProcessor,
	AutoModelForCausalLM,
	AutoProcessor,
	AutoTokenizer,
	Qwen2_5_VLForConditionalGeneration,
	Qwen2_5OmniForConditionalGeneration,
	Qwen2_5OmniProcessor,
	Qwen2AudioForConditionalGeneration,
	Qwen3OmniMoeForConditionalGeneration,
	Qwen3OmniMoeProcessor,
	Qwen3VLForConditionalGeneration,
	SwinModel,
	VoxtralForConditionalGeneration,
	enable_full_determinism,
)
from transformers import logging as hf_logging

enable_full_determinism(seed=42)

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)
os.environ['HF_HUB_VERBOSITY'] = 'error'
hf_logging.set_verbosity_error()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cuda_devices = ','.join([str(f) for f in select_devices(format='index', max_count=2, min_free_memory='23GiB')])
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
print(f'\n==> using CUDA devices: {cuda_devices}')


class OmniM:
	def __init__(self, args):
		print(f'initializing OmniM: {args.model_name}...')
		self.model_name = args.model_name
		model_class, processor_class = Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
		if self.model_name == 'Qwen/Qwen3-Omni-30B-A3B-Instruct':
			model_class, processor_class = Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
		self.model = model_class.from_pretrained(
			self.model_name,
			dtype=torch.bfloat16,
			device_map='auto',
			attn_implementation='flash_attention_2',
		)
		self.processor = processor_class.from_pretrained(args.model_name)
		if self.processor.tokenizer.pad_token_id is None:
			self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
			self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
		self.processor.tokenizer.padding_side = 'left'
		self.model.disable_talker()
		self.system_prompt = system_prompts[args.system_prompt_id]
		self.generation_prompt = generation_prompts[args.generation_prompt_id]
		self.target_ids = [self.processor.tokenizer(label, add_special_tokens=False).input_ids[0] for label in ['NO', 'YES']]
		print('...complete.\n')

	def create_conversation(self, sample):
		conversation = [{'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]}]
		content = []
		if sample['A'] is not None:
			content.append({'type': 'audio', 'audio': sample['A']})
		if sample['V'] is not None:
			content.append({'type': 'video', 'video': sample['V']})
		content.append({
			'type': 'text',
			'text': self.generation_prompt['with_text'].format(sample['L'])
			if sample['L'] is not None
			else self.generation_prompt['without_text'],
		})
		conversation.append({'role': 'user', 'content': content})
		return conversation

	def construct_input(self, raw_data):
		return [self.create_conversation(d) for d in raw_data]

	def get_model_output(self, model_input):
		text = self.processor.apply_chat_template(model_input, add_generation_prompt=True, tokenize=False)
		audios, _, videos = process_mm_info(model_input, use_audio_in_video=False)
		inputs = self.processor(
			text=text, audio=audios, videos=videos, return_tensors='pt', padding=True, use_audio_in_video=False
		)
		inputs = inputs.to(self.model.device).to(self.model.dtype)
		outputs = self.model.generate(
			**inputs,
			use_audio_in_video=False,
			return_audio=False,
			max_new_tokens=64,
			return_dict_in_generate=True,
			output_logits=True,
		)
		if self.model_name == 'Qwen/Qwen3-Omni-30B-A3B-Instruct':
			outputs = outputs[0]
		generated_ids, last_token_logits = outputs.sequences, outputs.logits[0]
		input_lengths = inputs.input_ids.shape[1]
		first_generated_token_ids = generated_ids[:, input_lengths].cpu().tolist()
		clipped_generated_ids = [
			output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
		]
		texts = self.processor.batch_decode(
			clipped_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)
		for idx, v in enumerate(first_generated_token_ids):
			assert v in self.target_ids, f'[ERROR] INVALID MODEL OUTPUT: {texts[idx]}, MODEL INPUT: {model_input[idx]}'
		target_logits = last_token_logits[:, self.target_ids]
		probabilities = torch.nn.functional.softmax(target_logits, dim=1).cpu().tolist()
		return [1 if 'YES' in t else 0 for t in texts], probabilities, texts


class LM:
	def __init__(self, args):
		print(f'initializing LM: {args.model_name}...')
		self.model = AutoModelForCausalLM.from_pretrained(
			args.model_name, dtype=torch.bfloat16, device_map='auto', attn_implementation='flash_attention_2'
		)
		self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
		self.tokenizer.padding_side = 'left'
		self.system_prompt = system_prompts[args.system_prompt_id]
		self.generation_prompt = generation_prompts[args.generation_prompt_id]
		self.model_name = args.model_name
		self.target_ids = [self.tokenizer(label, add_special_tokens=False).input_ids[0] for label in ['NO', 'YES']]
		print('...complete.\n')

	def create_conversation(self, sample):
		conversation = []
		if self.model_name in [
			'Qwen/Qwen2.5-7B-Instruct',
			'Qwen/Qwen3-4B-Instruct-2507',
			'meta-llama/Meta-Llama-3.1-8B-Instruct',
		]:
			conversation.append({'role': 'system', 'content': self.system_prompt})
		conversation.append({
			'role': 'user',
			'content': self.generation_prompt['with_text'].format(sample['L'])
			if sample['L'] is not None
			else self.generation_prompt['without_text'],
		})
		return conversation

	def construct_input(self, raw_data):
		return [self.create_conversation(d) for d in raw_data]

	def get_model_output(self, model_input):
		if self.model_name in [
			'Qwen/Qwen2.5-7B-Instruct',
			'Qwen/Qwen3-4B-Instruct-2507',
			'meta-llama/Meta-Llama-3.1-8B-Instruct',
		]:
			texts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in model_input]
			model_inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.model.device)
			generated_ids = self.model.generate(**model_inputs, max_new_tokens=64)
			input_lengths = model_inputs.input_ids.shape[1]
			first_generated_token_ids = generated_ids[:, input_lengths].cpu().tolist()
			generated_ids = [
				output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
			]
			texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
			for idx, v in enumerate(first_generated_token_ids):
				assert v in self.target_ids, f'[ERROR] INVALID MODEL OUTPUT: {texts[idx]}, MODEL INPUT: {model_input[idx]}'
			with torch.no_grad():
				prepared_inputs = self.model.prepare_inputs_for_generation(**model_inputs)
				outputs = self.model(**prepared_inputs)
				last_token_logits = outputs.logits[:, -1, :]
				target_logits = last_token_logits[:, self.target_ids]
				probabilities = torch.nn.functional.softmax(target_logits, dim=1).cpu().tolist()
			return [1 if 'YES' in t else 0 for t in texts], probabilities, texts


class ALM:
	def __init__(self, args):
		print(f'initializing ALM: {args.model_name}...')
		print(f'\tloading model and processor: {args.model_name}...', end='')
		if args.model_name == 'Qwen/Qwen2-Audio-7B-Instruct':
			self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
				args.model_name, dtype=torch.bfloat16, device_map='auto'
			)
		elif args.model_name == 'mistralai/Voxtral-Mini-3B-2507':
			self.model = VoxtralForConditionalGeneration.from_pretrained(
				args.model_name,
				dtype=torch.bfloat16,
				device_map='auto',
				attn_implementation='flash_attention_2',
				trust_remote_code=True,
			)
		self.processor = AutoProcessor.from_pretrained(args.model_name)
		self.system_prompt = system_prompts[args.system_prompt_id]
		self.generation_prompt = generation_prompts[args.generation_prompt_id]
		self.target_ids = [
			self.processor.tokenizer(label, add_special_tokens=False).input_ids[0]
			for label in ['NO', 'YES', 'Yes', 'No', 'yes', 'no']
		]
		self.model_name = args.model_name
		print('...complete.\n')

	def create_conversation(self, sample):
		conversation = []
		content = [
			{
				'type': 'text',
				'text': self.generation_prompt['with_text'].format(sample['L'])
				if sample['L'] is not None
				else self.generation_prompt['without_text'],
			}
		]
		if self.model_name == 'Qwen/Qwen2-Audio-7B-Instruct':
			conversation.append({'role': 'system', 'content': self.system_prompt})
			if sample['A'] is not None:
				content.append({'type': 'audio', 'audio_url': sample['A']})
		elif self.model_name == 'mistralai/Voxtral-Mini-3B-2507':
			content = [{'type': 'audio', 'path': sample['A']}] if sample['A'] is not None else []
		conversation.append({'role': 'user', 'content': content})
		return conversation

	def construct_input(self, raw_data):
		return [self.create_conversation(d) for d in raw_data]

	def get_model_output(self, model_input):
		if self.model_name == 'Qwen/Qwen2-Audio-7B-Instruct':
			texts = [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in model_input]
			audios = []
			for conversation in model_input:
				for message in conversation:
					if isinstance(message['content'], list):
						for ele in message['content']:
							if ele['type'] == 'audio':
								audios.append(
									librosa.load(
										BytesIO(open(ele['audio_url'], 'rb').read()),
										sr=self.processor.feature_extractor.sampling_rate,
									)[0]
								)

			if len(audios) == 0:
				inputs = self.processor(text=texts, return_tensors='pt', padding=True)
			else:
				inputs = self.processor(text=texts, audio=audios, return_tensors='pt', padding=True)
			inputs = inputs.to(self.model.device)
			outputs = self.model.generate(**inputs, max_new_tokens=64, return_dict_in_generate=True, output_logits=True)
			generated_ids, last_token_logits = outputs.sequences, outputs.logits[0]
			input_lengths = inputs.input_ids.shape[1]
			first_generated_token_ids = generated_ids[:, input_lengths].cpu().tolist()
			texts = self.processor.batch_decode(
				generated_ids[:, inputs.input_ids.size(1) :], skip_special_tokens=True, clean_up_tokenization_spaces=False
			)
			for idx, v in enumerate(first_generated_token_ids):
				if v not in self.target_ids:
					print(f'[ERROR] INVALID MODEL OUTPUT: {texts[idx]}, MODEL INPUT: {model_input[idx]}')
			target_logits = last_token_logits[:, self.target_ids]
			probabilities = torch.nn.functional.softmax(target_logits, dim=1).cpu().tolist()
			return [1 if 'yes' in t.lower() else 0 for t in texts], probabilities, texts
		elif self.model_name == 'mistralai/Voxtral-Mini-3B-2507':
			inputs = self.processor.apply_chat_template(model_input)
			inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
			outputs = self.model.generate(**inputs, max_new_tokens=64, return_dict_in_generate=True, output_logits=True)
			generated_ids, last_token_logits = outputs.sequences, outputs.logits[0]
			input_lengths = inputs.input_ids.shape[1]
			first_generated_token_ids = generated_ids[:, input_lengths].cpu().tolist()
			texts = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
			for idx, v in enumerate(first_generated_token_ids):
				assert v in self.target_ids, f'[ERROR] INVALID MODEL OUTPUT: {texts[idx]}, commentary: {model_input[idx]}'
			target_logits = last_token_logits[:, self.target_ids]
			probabilities = torch.nn.functional.softmax(target_logits, dim=1).cpu().tolist()
			return [1 if 'YES' in t else 0 for t in texts], probabilities, texts


class VLM:
	def __init__(self, args):
		print(f'initializing VLM: {args.model_name}...')
		model_class = (
			Qwen2_5_VLForConditionalGeneration
			if args.model_name == 'Qwen/Qwen2.5-VL-7B-Instruct'
			else Qwen3VLForConditionalGeneration
		)
		self.model = model_class.from_pretrained(
			args.model_name,
			dtype=torch.bfloat16,
			device_map='auto',
			attn_implementation='flash_attention_2',
			trust_remote_code=True,
		)
		self.processor = AutoProcessor.from_pretrained(args.model_name)
		self.system_prompt = system_prompts[args.system_prompt_id]
		self.generation_prompt = generation_prompts[args.generation_prompt_id]
		self.target_ids = [self.processor.tokenizer(label, add_special_tokens=False).input_ids[0] for label in ['NO', 'YES']]
		print('...complete.\n')

	def create_conversation(self, sample):
		conversation = [{'role': 'system', 'content': self.system_prompt}]
		content = [{'type': 'video', 'video': sample['V']}] if sample['V'] is not None else []
		content.append({
			'type': 'text',
			'text': self.generation_prompt['with_text'].format(sample['L'])
			if sample['L'] is not None
			else self.generation_prompt['without_text'],
		})
		conversation.append({'role': 'user', 'content': content})
		return conversation

	def construct_input(self, raw_data):
		return [self.create_conversation(d) for d in raw_data]

	def get_model_output(self, model_input):
		texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in model_input]
		image_inputs, video_inputs, video_kwargs = process_vision_info(model_input, return_video_kwargs=True)
		inputs = self.processor(
			text=texts,
			images=image_inputs,
			videos=video_inputs,
			padding=True,
			return_tensors='pt',
			**video_kwargs,
		)
		inputs = inputs.to(self.model.device)
		outputs = self.model.generate(**inputs, max_new_tokens=64, return_dict_in_generate=True, output_logits=True)
		generated_ids, last_token_logits = outputs.sequences, outputs.logits[0]
		input_lengths = inputs.input_ids.shape[1]
		first_generated_token_ids = generated_ids[:, input_lengths].cpu().tolist()
		generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
		texts = self.processor.batch_decode(
			generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)
		for idx, v in enumerate(first_generated_token_ids):
			if v not in self.target_ids:
				print(f'[ERROR] INVALID MODEL OUTPUT: {texts[idx]}, MODEL INPUT: {model_input[idx]}')
		target_logits = last_token_logits[:, self.target_ids]
		probabilities = torch.nn.functional.softmax(target_logits, dim=1).cpu().tolist()
		return [1 if 'YES' in t else 0 for t in texts], probabilities, texts


class Baseline:
	def __init__(self, args):
		print(f'initializing baseline ({args.modalities})...')
		self.modalities = args.modalities
		print('...complete.\n')

	def extract_mfcc(self, filepath, sr=44100, n_mfcc=20):
		y, _ = librosa.load(filepath, sr=sr)
		mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
		return np.mean(mfcc.T, axis=0)

	def extract_swin_features(self, video_path, sample_rate=1, batch_size=8):
		processor = AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
		model = SwinModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224').to(device)
		model.eval()
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps == 0:
			return None
		frame_interval, frames, frame_idx = int(fps * sample_rate), [], 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			if frame_idx % frame_interval == 0:
				frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
			frame_idx += 1
		cap.release()

		if not frames:
			return None
		all_features = []
		for i in range(0, len(frames), batch_size):
			batch = frames[i : i + batch_size]
			inputs = processor(images=batch, return_tensors='pt', padding=True).to(device)
			with torch.no_grad():
				outputs = model(**inputs)
				cls_embeddings = outputs.last_hidden_state[:, 0, :]
				all_features.append(cls_embeddings.cpu().numpy())
		all_features = np.vstack(all_features)
		return np.mean(all_features, axis=0)

	def get_features(self, X_train, X_test):
		if self.modalities == 'A':
			return (
				[self.extract_mfcc(fp) for fp in tqdm(X_train)],
				[self.extract_mfcc(fp) for fp in tqdm(X_test)],
			)
		elif self.modalities == 'V':
			return (
				[self.extract_swin_features(fp) for fp in tqdm(X_train)],
				[self.extract_swin_features(fp) for fp in tqdm(X_test)],
			)
		elif self.modalities == 'L':
			self.vectorizer = CountVectorizer(ngram_range=(1, 4))
			return (
				self.vectorizer.fit_transform([json.load(open(fp))['global'] for fp in X_train]),
				self.vectorizer.transform([json.load(open(fp))['global'] for fp in X_test]),
			)
		raise NotImplementedError

	def get_features_test(self, X_test):
		if self.modalities == 'A':
			return [self.extract_mfcc(fp) for fp in tqdm(X_test)]
		elif self.modalities == 'V':
			return [self.extract_swin_features(fp) for fp in tqdm(X_test)]
		elif self.modalities == 'L':
			return self.vectorizer.transform([json.load(open(fp))['global'] for fp in X_test])
		raise NotImplementedError

	def train_test_eval(self, X_train, X_test, y_train, y_test, save_as):
		X_train_feats, X_test_feats = self.get_features(X_train, X_test)
		clf = LogisticRegression(verbose=1)
		clf.fit(X_train_feats, y_train)
		joblib.dump(clf, f'{save_as}_model.joblib')
		if self.modalities == 'L':
			joblib.dump(self.vectorizer, f'{save_as}_vectorizer.joblib')
		y_pred = clf.predict(X_test_feats)
		f1, accuracy, cf_matrix = f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)
		print(f'\n\tF1-score: {f1}, Accuracy: {accuracy}, confusion matrix: {cf_matrix}', sep='\n\t')

	def get_model_responses(self, X_test, load_from):
		if self.modalities == 'L':
			self.vectorizer = joblib.load(f'{load_from}_vectorizer.joblib')
		X_test_feats = self.get_features_test(X_test)
		clf = joblib.load(f'{load_from}_model.joblib')
		return clf.predict_proba(X_test_feats).tolist()
