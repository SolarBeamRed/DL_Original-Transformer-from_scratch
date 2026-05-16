## Self Notes

---

### Multihead Attention Class

<u>Purpose</u>: Receives static embeddings and generates contextual embeddings

Conceptually, multiple weight matrices are used for multihead attention. During Implementation however, single set of large Weight matrices are used, which basically represent the multiple different weight matrices concatenated together. This helps with simultaneous attention calculation

`view()` function is used to return a new tensor with same values but different shape. This is used because dimensions change when going from single head to multiple heads in the `split_heads()` function

<u>Dimensions</u>:

- Input dimensions received by `forward()`:
  ```[batch_size, seq_length, d_model]```

- After `split_heads()`:
  ```[batch_size, num_heads, seq_length, d_k]```

- Dimensions received by `calculate_attention()`: Same as dimensions after split_heads

- After `combine_heads()`:
  ```[batch_size, seq_length, d_model]```

Final dimensions returned after calculating attention: Same as received during forward, i.e.
```[batch_size, seq_length, d_model]```

---

### Positional Feed Forward Class

Nothing special, just a normal Feed Forward block with ReLU activation after the first layer. Helps performance by introducing potentially necessary non-linearity

- Architecture: 2 layers

- First layer structure:
  ```[d_model, d_ff]```

- Second layer structure:
  ```[d_ff, d_model]```

- Activation: ReLU on first layer

Second layer's size being `d_model` ensures that dimensionality remains same, so further operations such as Add and Normalize are still implementable with simplicity

---

### Positional Encoding Class

`max_seq_length` does NOT mean seq_length of batch or examples. It's an arbitrary value which basically says:
 "how many positions should I keep my pe calculated?". 
`pe` is the table of positional encoding values for each dimension and for each position

When `x` is sent to get positional encoding added, only pe terms until `x`'s length are chosen and added automatically

#### Dimensions

- pe: ```[max_seq_length, d_model]```

- position: ```[max_seq_length, 1]```

- div_term: ```[d_model/2]``` (later broadcast to each position automatically)

- Final pe: ```[1, max_seq_length, d_model]```

`pe` is finally unsqueezed at `axis=0` to make it compatible with batch calculations. The same `pe` can be broadcast across all examples in the batch, since positional encoding terms to be added remain same

---

### Encoder Class

Simple encoder block with one Multihead Attention block and one Feed Forward block, with Add and Norm steps where necessary. Dropout is also used as a way to regularize model

Two different `norm` objects are used, since there is need to maintain two separate γ and β values

Dropout randomly turns off some dimensions of attention outputs and feed forward outputs during the "Add" step before normalization. This seems to help the model generalize better and prevent "memorizing" patterns instead of understanding

Queries, keys and values are all calculated from input in the Encoder block

Layer normalization is used, since BatchNorm takes even padded tokens into consideration while normalizing, which is not ideal. LayerNorm prevents padded tokens from affecting normalization of other token embeddings, which is preferred over BatchNorm's behavior

---

### Decoder Class

Decoder block containing one Masked Multihead Attention block, one Cross Attention block and one Feed Forward block, with respective Add and Norm steps where required. Dropout is used in similar fashion as in Encoder block

#### Masks

- src_mask: padding mask

- tgt_mask:  combination of causal mask (mask used in masked multihead/"nopeak" mask) and padding mask

Mask generation seen in the Transformer section

Cross attention block uses decoder's `attn_outputs` for Query, and `enc_outputs` for Key and Value embeddings

---

### Transformer Class Input and Mask Generation

Contains the whole architecture, including blocks for generating static embeddings for Encoder and Decoder, the Encoder-Decoder block themselves and mask generation

#### Format

- Inputs:<br>
  - src: ```[batch_size, src_seq_length]```
  - tgt: ```[batch_size, tgt_seq_length]```

- src_mask: ```[batch_size, 1, 1, seq_length]```

- tgt_mask (before integrating with causal): ```[batch_size, 1, seq_length, 1]```

- causal_mask: ```[1, seq_length, seq_length]```

- Final tgt_mask: ```[batch_size, 1, seq_length, seq_length]```

  after AND operation with causal mask

Padding masks are both boolean, with 'True' values in non-padded indices, and 'False' for padded token indices
Causal mask is a square matrix of shape: [1, seq_length, seq_length] with upper triangle elements being 0s, and the rest being 1s<br>

`triu()` with diagonal=1 returns a matrix with 1s in upper triangle, and 0s in remaining positions, which is then subtracted from 1 to get required causal mask

3×3 example:

```text
1 - [[0 1 1]         [[1 0 0]
     [0 0 1]   --->   [1 1 0]
     [0 0 0]]         [1 1 1]]
```

Then integrated with padding mask using AND operation

Decoder outputs are then passed to a final linear layer with dimensions: ```[d_model, tgt_vocab_size]``` to get logits for the next token that is most likely

___

### Testing inference without testing
- Arbitrarily chose src_vocab_size and tgt_vocab_size. I went with 1500. 
- Create a Transformer object named model, optionally send it to GPU
- Create toy src and tgt. I chose batch_size to be 16, src_seq_length to be 12 and tgt_seq_length to be 14
- Mock padding by arbitrarily setting some examples of src and tgt to contain padding tokens(0) in the end
- Pass src and tgt to Transformer model
- Verify if output shapes is same as expected. The output size should be `[batch_size, tgt_seq_length, tgt_vocab_size]`, which is suppsed to be [16, 14, 1500]