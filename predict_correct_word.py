import torch
from transformers import BertTokenizer, BertForMaskedLM
from difflib import get_close_matches

# Load the trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("spell_corrector_bert")
model = BertForMaskedLM.from_pretrained("spell_corrector_bert")

# Load vocabulary from tokenizer
vocab = tokenizer.get_vocab()
inv_vocab = {v: k for k, v in vocab.items()}

# Function for hybrid prediction
def correct_word(sentence, wrong_word, top_k=5, threshold=0.8):
    # Replace the wrong word with [MASK]
    masked_sentence = sentence.replace(wrong_word, '[MASK]')

    # Tokenize input
    inputs = tokenizer(masked_sentence, return_tensors="pt")

    # Predict the masked word
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get the index of the [MASK] token
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1].item()

    # Extract top-k predictions
    logits = predictions[0, mask_token_index]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k_indices = torch.topk(probs, top_k * 10).indices  # Get more words initially

    # Decode top predictions
    suggestions = [tokenizer.decode(idx).strip() for idx in top_k_indices]

    # Apply edit distance filtering
    close_matches = get_close_matches(wrong_word, suggestions, n=top_k, cutoff=threshold)
    return close_matches if close_matches else suggestions[:top_k]

# Function for autocomplete
def autocomplete(sentence, top_k=5):
    # Append [MASK] at the end for prediction
    masked_sentence = sentence + ' [MASK]'

    # Tokenize input
    inputs = tokenizer(masked_sentence, return_tensors="pt")

    # Predict the masked word
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get the index of the [MASK] token
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1].item()

    # Extract top-k predictions
    logits = predictions[0, mask_token_index]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k_indices = torch.topk(probs, top_k).indices

    # Decode top predictions
    suggestions = [tokenizer.decode(idx) for idx in top_k_indices]
    return suggestions

# Example usage
if __name__ == "__main__":
    # Spell correction example
    #"liverpool is the champin of the premier league."
    #I am going to the stoe to buy some milk.
    #I will file a lwasuit against you.
    #The biggest city in Turkey is Istagnbul
    sentence = "The biggest city in Turkey is Istagnbul"
    wrong_word = "Istagnbul"
    suggestions = correct_word(sentence, wrong_word)
    print(f"Suggestions for '{wrong_word}': {suggestions}")

    # Autocomplete example
    #"In the morning kids goes to"
    
    incomplete_sentence = "Capital of Turkey is"
    autocomplete_suggestions = autocomplete(incomplete_sentence)
    print(f"Autocomplete suggestions: {autocomplete_suggestions}")
