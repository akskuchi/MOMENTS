system_prompts = {
	's1': 'You are an experienced commentator for soccer matches, who is well versed with the knowledge of the game.',
	's2': 'You are an expert of soccer games, who is well versed with the knowledge of the game, and capable of recognizing game highlights.',
}

generation_prompts = {
	'g1': {
		'with_text': 'Classify whether this is an important moment of the game that should be included in the highlights. Here is the commentary associated with this moment: "{}". Respond only with a "YES" or a "NO".',
		'without_text': 'Classify whether this is an important moment of the game that should be included in the highlights. Respond only with a "YES" or a "NO".',
	},
	'g2': {
		'with_text': 'Is this an important moment of the game that is worthy of inclusion in the highlights reel? This is the transcription of the commentary: "{}". Respond only with a "YES" or a "NO".',
		'without_text': 'Is this an important moment of the game that is worthy of inclusion in the highlights reel? Respond only with a "YES" or a "NO".',
	},
}
