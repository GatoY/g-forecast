# Seq2Seq-with-attention
Seq2Seq model with attention to translate German to English.

## Requirements

- Python 3.6
- Pytorch 0.4.0
- spacy
- torchtext 0.3.1

## Dataset

- Multi30k from torchtext.Dataset

## Model

- Encoder: Bidirectional GRU
- Decoder: GRU with Attention Mechanism
- Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## References

- [@keon/seq2seq](https://github.com/keon/seq2seq)
- Pytorch Tutorials: [Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

