# Transformer Implementation From Scratch Using PyTorch
### Overview
Building the Transformer model from the 2017 "Attention Is All You Need" Paper using PyTorch
without any high level transformer libraries. 
The repo contains the notebook containing model architecture code, and a markdown file 
with notes about each class's working. I did not train the model on any datasets since 
my hardware would limit me from training on large corpora and struggle to provide high 
performance

___
### Features Implemented
- Multi-Head Self Attention
- Masked Multi-Head Attention
- Encoder-Decoder Cross Attention
- Positional Encoding
- Position-wise Feed Forward Networks
- Residual Connections
- Layer Normalization
- Padding + Causal Masks
- Encoder and Decoder stacks
- Dropout
- End-to-end forward pass verification

___
### Architecture
Encoder stack is made of multiple Encoder blocks, each block consisting: <br>
1. MultiHead Attention Block
2. Add and Norm
3. Feed Forward Block
4. Add and Norm

Decoder stack is made of multiple Decoder blocks, each block consisting: <br>
1. Masked MultiHead Attention Block
2. Add and Norm
3. Cross Attention Block
4. Add and Norm
5. Feed Forward Block
6. Add and Norm

Decoder stack's output is then fed to a final linear layer with size `[d_model, tgt_vocab_size]` 
to obtain logits for next token's prediction

User can choose the model's hyperparameters such as number of Encoder and Decoder blocks 
per Encoder or Decoder stack, model's dimensionality, Feed Forward network's size etc.
<br> An example of the model on toy src_vocab and tgt_vocab would be:
```text
model = Transformer(
    src_vocab_size=1500,
    tgt_vocab_size=1500,
    num_layers=4,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    max_seq_length=50,
    dropout_p=0.1
)
```

___
### Tensor Dimensions
These were perhaps the most confusing part of the implementation. To me, matching the dimensions 
of different tensor inputs to different functions was the tricky part. So I made notes 
about the formats and dimensions of tensors in each class <br>
For example, the MultiHead Attention class's format is as follows: 
- Input dimensions received by `forward()`:
  ```[batch_size, seq_length, d_model]```

- After `split_heads()`:
  ```[batch_size, num_heads, seq_length, d_k]```

- Dimensions received by `calculate_attention()`: Same as dimensions after split_heads

- After `combine_heads()`:
  ```[batch_size, seq_length, d_model]```

Final dimensions returned after calculating attention: Same as received during forward, i.e.
```[batch_size, seq_length, d_model]```

I have included dimensions of each class in `self_notes.md`. Feel free to check it out

___
### Masks

- src_mask: padding mask
- tgt_mask:  combination of causal mask (mask used in masked multihead/"nopeak" mask) and padding mask

#### Dimensions
- src_mask: ```[batch_size, 1, 1, seq_length]```

- tgt_mask (before integrating with causal): ```[batch_size, 1, seq_length, 1]```

- causal_mask: ```[1, seq_length, seq_length]```

- Final tgt_mask: ```[batch_size, 1, seq_length, seq_length]```

  after AND operation with causal mask

Padding masks are both boolean, with 'True' values in non-padded indices, and 'False' for padded token indices
Causal mask is a square matrix of shape: [1, seq_length, seq_length] with upper triangle elements being 0s, and the rest being 1s<br>

`triu()` with diagonal=1 returns a matrix with 1s in upper triangle, and 0s in remaining positions, which is then subtracted from 1 to get required causal mask

___
### Forward Pass Verification
The architecture was verified through successful end-to-end forward passes 
using mock source/target batches with artificial padding. As mentioned previously, I refrained from training the model
since my RTX 3050 with 4GB VRAM would really struggle in training models large enough to obtain good performance<br><br>

Toy data used for test:<br>
`src = torch.randint(1, 1500, (16, 12)).to(device)`<br>
`tgt = torch.randint(1, 1500, (16, 14)).to(device)`<br><br>

Mock padding:<br>
`src[0, -3:] = 0`<br>
`tgt[2, -2:] = 0`<br><br>

Expected and received output size:<br>
`torch.Size([16, 14, 1500])`

___
### Limitations
This project focuses on implementing and understanding Transformer architecture internals. Dataset training, 
tokenization pipelines, and optimization for large-scale NLP tasks were intentionally left outside the scope.

___
### Running the Notebook
Just make sure the only required library PyTorch is installed, along with Jupyterlab/Jupyter-notebook. Then, just run all the cells
 in the notebook in order

___
### Extra
Detailed notes about implemented classes and their input-output formats available in `self_notes.md`