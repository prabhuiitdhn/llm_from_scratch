"""
Production-grade tokenizer usage with pre-trained vocabularies.

This script demonstrates:
- Loading pre-built tokenizers from Hugging Face (no vocab building needed)
- Why pre-trained tokenizers are critical for fine-tuning
- Real-world use cases: instruction tuning, LoRA, inference
- Vocabulary consistency across train/validation/inference
- Handling special tokens for domain-specific tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TokenizedExample:
	"""Single tokenized example for training/validation/inference."""

	text: str
	input_ids: List[int]
	attention_mask: List[int]
	token_count: int


def example_1_basic_pretrained_tokenizer() -> None:
	"""
	Example 1: Load and use a pre-trained tokenizer.
	
	This is what 90% of fine-tuning projects do — no custom vocabulary building.
	"""
	print("\n" + "=" * 72)
	print("Example 1: Basic Pre-trained Tokenizer (Meta-Llama-2-7b)")
	print("=" * 72)

	try:
		from transformers import AutoTokenizer

		# Production approach: Load vocabulary + special tokens from pre-trained model
		tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

		print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
		print(f"✓ Special tokens: {tokenizer.special_tokens_map}")
		print(f"✓ Padding token: '{tokenizer.pad_token}'")
		print(f"✓ EOS token: '{tokenizer.eos_token}'")

		# Tokenize example text
		text = "Fine-tuning a large language model requires careful vocabulary handling."
		encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

		print(f"\nText: {text}")
		print(f"Input IDs: {encoded['input_ids'].tolist()}")
		print(f"Attention mask: {encoded['attention_mask'].tolist()}")
		print(f"Token count: {len(encoded['input_ids'][0])}")

		# Decode back to verify consistency
		decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
		print(f"Decoded: {decoded}")

	except ImportError:
		print("⚠ Transformers library not installed. Install with: pip install transformers")


def example_2_instruction_tuning_format() -> None:
	"""
	Example 2: Tokenizing instruction-tuning format (common in fine-tuning).
	
	Instruction tuning adds special tokens to format Q&A or task data.
	"""
	print("\n" + "=" * 72)
	print("Example 2: Instruction Tuning Format (with special tokens)")
	print("=" * 72)

	try:
		from transformers import AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

		# Common instruction-tuning format
		instruction_template = """\
[INST] What is fine-tuning? [/INST]

Fine-tuning is the process of adapting a pre-trained model by training it on task-specific data."""

		# Tokenize with special formatting
		encoded = tokenizer(
			instruction_template,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=256,
		)

		print(f"Instruction text:\n{instruction_template}\n")
		print(f"Input IDs (first 30): {encoded['input_ids'][0][:30].tolist()}")
		print(f"Attention mask (first 30): {encoded['attention_mask'][0][:30].tolist()}")
		print(f"Total tokens: {len(encoded['input_ids'][0])}")

		# Verify encoding stability (train vs inference consistency)
		re_encoded = tokenizer(instruction_template, return_tensors="pt")
		print(f"\n✓ Consistent encoding: {torch.allclose(encoded['input_ids'], re_encoded['input_ids'])}")

	except ImportError:
		print("⚠ Transformers library not installed.")


def example_3_batch_tokenization_for_training() -> None:
	"""
	Example 3: Batch tokenization for training loops (real-world scenario).
	
	Shows how pre-built vocabulary enables efficient batch processing.
	"""
	print("\n" + "=" * 72)
	print("Example 3: Batch Tokenization for Training")
	print("=" * 72)

	try:
		from transformers import AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Smaller vocab, faster demo

		# Simulated training batch
		texts = [
			"LoRA fine-tuning adapts only low-rank projections of weight matrices.",
			"QLoRA quantizes the model to 4-bit for memory-efficient training.",
			"PEFT provides parameter-efficient alternatives to full model tuning.",
		]

		# Batch tokenize with unified padding
		batch_encoded = tokenizer(
			texts,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=64,
		)

		print(f"Batch size: {len(texts)}")
		print(f"Vocab size (GPT-2): {tokenizer.vocab_size}")
		print(f"Max sequence length in batch: {batch_encoded['input_ids'].shape[1]}")

		for i, text in enumerate(texts):
			print(f"\n  Example {i + 1}:")
			print(f"    Text: {text[:50]}...")
			print(f"    Tokens: {batch_encoded['input_ids'][i][:15].tolist()}...")
			print(f"    Non-padding count: {batch_encoded['attention_mask'][i].sum().item()}")

	except ImportError:
		print("⚠ Transformers library not installed.")


def example_4_why_pretrained_vocabulary_matters() -> None:
	"""
	Example 4: Why pre-trained vocabularies are essential.
	
	Demonstrates consistency and optimization benefits.
	"""
	print("\n" + "=" * 72)
	print("Example 4: Why Pre-trained Vocabularies Matter")
	print("=" * 72)

	comparison = """
	CUSTOM VOCABULARY (Educational, from tokenization.py):
	  ✗ Built only from training data → rare/test tokens become <unk>
	  ✗ Inference time: Unknown domain terms fail silently
	  ✗ Not optimized for language patterns
	  ✗ Tiny vocab (thousands) vs production (tens of thousands)
	  ✗ Mismatch between train/val/inference behavior

	PRE-TRAINED VOCABULARY (Production, from Hugging Face):
	  ✓ Built from massive diverse datasets (billions of tokens)
	  ✓ Optimized with Byte-Pair Encoding (BPE) or SentencePiece
	  ✓ Vocab already tuned for language: 50k tokens (GPT-2) to 128k (Llama-2)
	  ✓ Handles rare domain terms gracefully via subword splitting
	  ✓ Consistent behavior across train/val/inference
	  ✓ Inference-time unknown words split into known subwords
	  ✓ Already paired with model weights — guaranteed compatibility

	REAL EXAMPLE:
	  Custom tokenizer on unseen domain term "RLHF":
	    → Tokens: ['R', 'L', 'H', 'F'] mapped to <unk>  (loses meaning)
	  
	  Pre-trained tokenizer (Llama-2):
	    → Tokens: ['R', 'LH', 'F'] or ['RLHF'] if in vocab
	                (preserves semantic chunks)
	"""
	print(comparison)


def example_5_loading_different_models() -> None:
	"""
	Example 5: Quick reference for loading different pre-trained tokenizers.
	
	Shows the diversity of available vocabularies for different tasks.
	"""
	print("\n" + "=" * 72)
	print("Example 5: Loading Different Pre-trained Tokenizers")
	print("=" * 72)

	examples = """
	LANGUAGE MODELS (General purpose):
	  from transformers import AutoTokenizer
	  
	  # Open-source models
	  tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
	  tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
	  tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
	  
	  # Smaller, faster models
	  tok = AutoTokenizer.from_pretrained("gpt2")
	  tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

	ENCODER-ONLY MODELS (BERT-style, classification):
	  tok = AutoTokenizer.from_pretrained("bert-base-uncased")
	  tok = AutoTokenizer.from_pretrained("roberta-base")

	MULTILINGUAL MODELS:
	  tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
	  tok = AutoTokenizer.from_pretrained("mBERT-base-multilingual-cased")

	DOMAIN-SPECIFIC MODELS:
	  tok = AutoTokenizer.from_pretrained("allenai/scibert-base-uncased")  # Scientific
	  tok = AutoTokenizer.from_pretrained("emilyalsentzer/clinicalBERT")   # Medical

	PROPRIETARY (via APIs or local):
	  # For GPT-3/GPT-4 — use tiktoken
	  import tiktoken
	  enc = tiktoken.encoding_for_model("gpt-4")
	  tokens = enc.encode("Your text here")

	KEY POINT:
	  All these tokenizers come with pre-built, optimized vocabularies.
	  You never need to run fit() or build your own for production use.
	"""
	print(examples)


def example_6_fine_tuning_workflow() -> None:
	"""
	Example 6: Complete fine-tuning workflow showing tokenizer role.
	
	Demonstrates the full pipeline: tokenization → model → loss → backprop.
	"""
	print("\n" + "=" * 72)
	print("Example 6: Complete Fine-tuning Workflow (Conceptual)")
	print("=" * 72)

	workflow = """
	FINE-TUNING WORKFLOW:
	
	1. LOAD PRE-TRAINED TOKENIZER + MODEL
	   ┌─────────────────────────────────────┐
	   │ from transformers import AutoTokenizer, AutoModelForCausalLM
	   │ tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
	   │ model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
	   └─────────────────────────────────────┘
	
	2. TOKENIZE TRAINING DATA (using pre-built vocabulary)
	   ┌─────────────────────────────────────┐
	   │ for batch in training_dataloader:
	   │   texts = batch['text']
	   │   encoded = tokenizer(texts, return_tensors="pt", padding=True,
	   │                        truncation=True, max_length=2048)
	   │   # encoded['input_ids'] ← Uses vocab of 32k tokens (no custom building!)
	   └─────────────────────────────────────┘
	
	3. FORWARD PASS
	   ┌─────────────────────────────────────┐
	   │ outputs = model(input_ids=encoded['input_ids'],
	   │                 attention_mask=encoded['attention_mask'],
	   │                 labels=encoded['input_ids'])
	   └─────────────────────────────────────┘
	
	4. BACKWARD PASS & UPDATE
	   ┌─────────────────────────────────────┐
	   │ loss = outputs.loss
	   │ loss.backward()
	   │ optimizer.step()
	   └─────────────────────────────────────┘
	
	5. INFERENCE (same tokenizer, same vocabulary)
	   ┌─────────────────────────────────────┐
	   │ input_text = "Your query here"
	   │ encoded = tokenizer(input_text, return_tensors="pt")
	   │ # CRITICAL: Same vocabulary ensures consistency!
	   │ output = model.generate(encoded['input_ids'], max_new_tokens=100)
	   │ response = tokenizer.decode(output[0])
	   └─────────────────────────────────────┘
	
	WHY THIS WORKS:
	  - Tokenizer vocabulary is immutable (built once, reused forever)
	  - Model weights are pre-optimized for that specific vocabulary
	  - No train/val/inference mismatch — same vocab throughout
	  - Unknown test tokens handled by BPE subword splitting
	"""
	print(workflow)


def run_all_examples() -> None:
	"""Run all production tokenizer examples."""
	print("\n" + "█" * 72)
	print("█" + " " * 70 + "█")
	print("█" + "  PRODUCTION TOKENIZERS: Pre-built Vocabularies for Fine-tuning".center(70) + "█")
	print("█" + " " * 70 + "█")
	print("█" * 72)

	# Note: Some examples require transformers library
	example_1_basic_pretrained_tokenizer()
	example_2_instruction_tuning_format()
	example_3_batch_tokenization_for_training()
	example_4_why_pretrained_vocabulary_matters()
	example_5_loading_different_models()
	example_6_fine_tuning_workflow()

	print("\n" + "=" * 72)
	print("SUMMARY")
	print("=" * 72)
	print("""
	KEY TAKEAWAYS:

	1. For production/fine-tuning: ALWAYS use pre-trained tokenizers
	   → No custom vocabulary building needed
	   → Vocabulary already optimized and tested

	2. Pre-trained tokenizers solve consistency problems:
	   → Train/val/inference all use identical vocabulary
	   → Rare/unseen tokens handled via subword decomposition
	   → No silent failures from <unk> tokens

	3. When to build custom vocabulary:
	   → RARELY — only research prototyping or specialized closed domains
	   → NOT for production
	   → NOT for fine-tuning existing models

	4. Quick reference:
	   from transformers import AutoTokenizer
	   tokenizer = AutoTokenizer.from_pretrained("model-name")
	   # That's it — vocabulary + special tokens come for free
	""")


if __name__ == "__main__":
	run_all_examples()
