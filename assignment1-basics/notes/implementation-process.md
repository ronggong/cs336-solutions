# Implementation Process

This assignment helped connect the mathematical descriptions of language-model components with the practical concerns of implementing them efficiently in PyTorch. The most important lessons were about finding the right data structures, making tensor operations readable, and estimating computational and memory costs.

## Byte-Pair Encoding

The most challenging part for me was implementing BPE and making it scale. The merge operation is inherently sequential: after choosing the best pair and merging it, the counts and neighboring pairs can change, so the next choice depends on the previous one. This means that the entire merge loop cannot simply be parallelized.

A straightforward implementation repeatedly scans all pair counts to find the maximum. That is expensive, especially when the vocabulary and training corpus become large. A better approach is to maintain a heap-based priority queue of candidate pairs. Since Python's `heapq` is a min-heap, the priority can be reversed so that the pair with the largest count is selected first. The pair representation needs a deterministic ordering, including a custom less-than comparison when the desired tie-breaking order is not provided automatically.

Heap entries can become stale after a merge changes the count of a pair. The implementation therefore needs to validate a candidate when it is popped and skip or refresh it if it no longer represents the current count. This is a general priority-queue technique: updates are often cheaper to add lazily than to remove old entries in place.

Another important optimization is to avoid scanning every pre-token from beginning to end after each merge. Only pre-tokens containing the modified pair, together with their affected neighbors, need to be updated. Keeping track of those affected tokens reduces unnecessary work substantially. The main idea is to update the local state that changed rather than recomputing the whole corpus after every merge.

BPE training on a larger corpus such as OpenWebText can take a long time in a naive Python implementation. Depending on the corpus, implementation, and hardware, it may take hours or longer without these optimizations. Efficient candidate selection and localized updates are therefore not just small improvements: they determine whether the implementation is practical on limited resources.

## Transformer Building Blocks

When implementing the Transformer, `einops` operations such as `rearrange` made tensor transformations much easier to read. Matrix multiplication often requires moving between representations such as

- `(batch, sequence, model dimension)`;
- `(batch, heads, sequence, head dimension)`; and
- flattened or paired dimensions for positional encoding.

Without named rearrangements, the same operations would require many manual `reshape`, `transpose`, and `view` calls. Those calls are harder to inspect and make it easier to introduce a silent shape error. Naming the dimensions directly makes the intended computation clearer.

The linear layer is another foundational component. It is used throughout a Transformer, including the projections that produce queries, keys, and values. Implementing it correctly provides a simple and reusable abstraction for many of the model's other modules.

### Pre-Norm and Post-Norm

For the small Transformer language model used in this assignment, the post-norm configuration reached a loss similar to the pre-norm configuration. This suggests that, at this model scale and training setup, the choice between applying normalization before or after the sublayer did not have a large effect on the final loss. This result should not be generalized automatically to deeper models or different optimization settings, where pre-norm is often preferred for more stable gradient flow.

The intuition is that pre-norm preserves an identity residual route. A pre-norm sublayer has the form $x_{l+1} = x_l + F_l(\operatorname{Norm}(x_l))$. During backpropagation, its derivative includes an additive identity term: $I + J_{F_l}J_{\operatorname{Norm}}$. The non-residual branch still includes the normalization Jacobian, so normalization keeps the input to attention or the feed-forward network in a controlled range. However, gradients also have a direct route through the $I$ term.

In post-norm, $x_{l+1} = \operatorname{Norm}(x_l + F_l(x_l))$, so the normalization Jacobian multiplies the entire layer derivative: $J_{\operatorname{Norm}}(I + J_{F_l})$. This means the residual route is also transformed at every layer. Across a deep stack, these transformations can compound, making optimization more sensitive. Pre-norm therefore does not remove normalization; it keeps normalization inside the sublayer branch while leaving the residual connection direct.

Another practical finding was that removing positional embeddings from the query and key projections made the loss noticeably worse. This indicates that positional information is still important for the attention mechanism even when the model is otherwise implemented correctly; without it, the model loses the ability to distinguish token order in a way that is critical to the learning signal in this small setup.

## Rotary Position Embeddings

RoPE initially looked more complicated than it turned out to be. At its core, it applies a two-dimensional rotation independently to each pair of features. The implementation consists of computing the position-dependent sine and cosine values, applying the rotation to each pair, and restoring the original tensor layout.

More concretely, for every token position $t$ and feature-pair index $i$, RoPE precomputes $\\cos(t\\theta_i)$ and $\\sin(t\\theta_i)$. These values define a $2 \\times 2$ rotation for that pair of query or key features. If the head dimension is $d_k$, there are $d_k / 2$ such feature pairs, each with a different frequency $\\theta_i$. The full conceptual transformation is a block-diagonal matrix, but the implementation directly rotates each pair rather than constructing that large sparse matrix.

Although RoPE receives absolute positions to select rotations, it produces relative-position information in the attention score. A query at position $t$ is rotated by $R_t$ and a key at position $s$ by $R_s$. Their dot product is $$(R_t q_t)^T(R_s k_s) = q_t^T R_t^T R_s k_s = q_t^T R_{s-t} k_s,$$ so the interaction depends on the offset $s-t$. In causal self-attention, a query at $t$ compares with keys from positions $0$ through $t$, and RoPE lets the attention mechanism use their distances directly. A uniform shift of all position IDs therefore leaves those relative distances unchanged, although positions must remain consecutive when keys are retained in an autoregressive cache.

The main practical difficulty is not the rotation formula itself but keeping the shapes and broadcasting rules correct. `rearrange` helps expose the paired feature dimension, while position indexing must produce sine and cosine tensors that broadcast over batch and sequence dimensions as intended.

## FLOPs Estimation

For FLOPs calculations, matrix multiplication is the main operation to count. The dimensions of the contracting axes determine the amount of work. Dimensions that are not involved in the contraction, such as batch or head dimensions, are multiplicative outer dimensions and should be counted once.

For example, for a batched matrix multiplication with shape

```text
(..., m, k) @ (..., k, n)
```

the number of scalar multiply operations is proportional to

```text
(... product of outer dimensions ...) * m * k * n.
```

If a multiply and an addition are counted as separate floating-point operations, the estimate is commonly multiplied by two. The important point is to avoid multiplying a shared batch dimension twice merely because it appears in both input tensors: the batch is one outer dimension of the operation.

## Memory Estimation

Adam requires more memory than a model's parameter count alone suggests. In addition to the parameters, training generally stores gradients and two optimizer states: the first moment and the second moment. If all of these tensors use the same dtype, this is approximately four parameter-sized tensors in total:

- parameters: $P$ elements;
- gradients: $P$ elements;
- first moment: $P$ elements; and
- second moment: $P$ elements.

The optimizer states alone are therefore about two parameter-sized tensors. The often-used estimate of roughly three times the parameter memory refers to the additional gradient and two Adam states beyond the parameters; the exact byte count depends on dtype and whether a separate higher-precision copy of the parameters is maintained.

Activations are another major part of training memory. Their exact size depends on the computation graph, model architecture, sequence length, and which intermediate values must be saved for backpropagation. Unlike the parameter and optimizer-state terms, activation memory usually scales directly with the batch size and also with the sequence length. This is why increasing the batch size can make memory usage grow quickly even when the model parameters are unchanged.

Overall, the assignment showed that making a model work is only the first step. Data structures such as heaps, localized updates, clear tensor rearrangements, and explicit compute and memory estimates are what make an implementation usable at larger scale.
