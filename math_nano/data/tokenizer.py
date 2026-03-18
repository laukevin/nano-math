"""GPT-2 BPE tokenizer utilities via tiktoken."""

import tiktoken

EOT_TOKEN = 50256


def get_tokenizer() -> tiktoken.Encoding:
    """Get the GPT-2 BPE tokenizer."""
    return tiktoken.get_encoding("gpt2")


def tokenize_document(text: str, tokenizer: tiktoken.Encoding | None = None) -> list[int]:
    """Tokenize a single document, appending EOT.

    Args:
        text: Raw text to tokenize.
        tokenizer: Optional pre-created tokenizer (avoids repeated init).

    Returns:
        List of token IDs with EOT appended.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    tokens = tokenizer.encode_ordinary(text)
    tokens.append(EOT_TOKEN)
    return tokens
