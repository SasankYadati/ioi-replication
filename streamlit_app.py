import streamlit as st
st.set_page_config(page_title="IOI Replication")
st.title("Replicating Interpretability in the Wild")
st.caption("We find a circuit in GPT-2 small that performs the Indirect Object Identification (IOI) task.")

st.markdown("[Original paper](https://arxiv.org/pdf/2211.00593)")
st.markdown("[Reference](https://arena3-chapter1-transformer-interp.streamlit.app/[1.4.1]_Indirect_Object_Identification)")

st.header("Setup")

st.subheader("Load GPT-2 small")
st.write("GPT-2 small is a 12 layer 80 million parameter transformer.")
st.code("""
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
""")
with st.expander("See explanation for `center_unembed`, `center_writing_weights` and `fold_ln`"):    
    st.markdown("On `center_unembed=True`: Softmax function applied on the logits is translation invariant, so we can \
                simplify things by setting the mean of the logits to be zero. This is in turn equivalent to setting the mean of every output vector \
                of the unembed matrix to zero.")
    st.markdown("On `center_writing_weights=True`: The mean of the residual stream vectors doesn't matter because LayerNorm is first applied whenever a component is trying to read \
                from the residual stream. So, we can make the mean zero by centering the output vectors of weights \
                that write to the residual stream.")
    st.markdown("On `fold_ln=True`: [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#layernorm) is non-linear and cannot be turned off during inference. \
                If the LayerNorm is followed by a linear layer, we can fold the weights and biases of the layer norm\
                into the weights and biases of the linear layer. What remains after this folding is the zeroing the mean and dividing by the norm. \
                We don't have to zero the mean if we do `center_writing_weights=True`. What remains is the division by norm of the vector, which in practice we treat it as a constant across runs. \
                We get away with this sometimes.")
    st.markdown("Reference - [TransformerLens](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html#transformer_lens.HookedTransformer.HookedTransformer.from_pretrained)")

st.subheader("Performance on IOI task")
st.write("We test the model on a prompt to check whether it can do the task.")
st.code("""
example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)
""")
st.write("Output:")
st.code("""
Tokenized prompt: ['<|endoftext|>', 'After', ' John', ' and', ' Mary', ' went', ' to', ' the', ' store', ',', ' John', ' gave', ' a', ' bottle', ' of', ' milk', ' to']
Tokenized answer: [' Mary']
""", language="markdown")


st.write("We now construct a small dataset of 8 prompts for our exploratory analysis. \
    We will later verify our findings from here using more general and comprehensive dataset.")
st.code("""prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" John", " Mary"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]

# Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
prompts = [
    prompt.format(name) 
    for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1] 
]
# Define the answers for each prompt, in the form (correct, incorrect)
answers = [names[::i] for names in name_pairs for i in (1, -1)]
# Define the answer tokens (same shape as the answers)
answer_tokens = t.concat([
    model.to_tokens(names, prepend_bos=False).T for names in answers
])

rprint(prompts)
rprint(answers)
rprint(answer_tokens)

table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

for prompt, answer in zip(prompts, answers):
    table.add_row(prompt, repr(answer[0]), repr(answer[1]))

rprint(table)
""")


st.write("Run the model on the above prompts to obtain the logits and also cache the activations.")
st.code("""
tokens = model.to_tokens(prompts, prepend_bos=True)
tokens = tokens.to(device)
original_logits, cache = model.run_with_cache(tokens)
""")

st.write("To evaluate the performance of the model we will measure the logit difference \
    between the indirect object (IO, correct answer) and the subject (S, wrong answer). \
    Eg: logit(Mary) - logit(John)")

st.code("""
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
) -> Float[Tensor, "*batch"]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    logits_final_pos = logits[:, -1, :]
    answer_logits = logits_final_pos.gather(dim=-1, index=answer_tokens)
    logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return logit_diff
    return logit_diff.mean(dim=0)
""")

st.image("logit_differences.jpg")

st.subheader("Initial Hypothesis")
st.write("It's useful to have some hypthosis of how the model might be doing the task before we \
         do any experiments. \
         My hypothesis: Consider the example 'When John and Mary went to the shops, John gave the bag to ->  Mary'\
         The model detects the second John token as a duplicate and a later attention head moves information of Mary to the last position \
         (where the next token prediction is made).")

st.header("Logit Attribution")

st.subheader("Direction Logit Attribution")
st.markdown("""Idea: work backwards starting from the model's output
* The central object of a transformer is the residual stream, which is the sum of each layer's output, token embedding and positional embedding
* The logits are approximately a linear function of the final residual stream value:
    - logits = Unembed(LayerNorm(final_residual_stream_value))
    - LayerNorm isn't technically linear so logits are technically not a linear function of the residual stream.
* Each attention layer's output can be broken down into sum of the output of each attention head in the layer
* Each MLP layer's output can be broken down into sum of the output of each neuron and a bias term.
* Therefore, we can decompose the logits into the sum of contributions of each component and look at which components contribute the most to the logit of the correct token
    - Components: attention head, mlp neuron, token embedding, position embedding""")

st.markdown("""
            We will use the logit differnce between the correct token (IO) and incorrect token (S) as this controls for different things the model might be doing. For example, the model might want to predict a pronoun instead of a name.
            Since we repeat each prompt twice for each possible IO name, we also control for any biases the model have in predicting some names more frequently than the others.
            
            $$ \\text{logit diff} = (x^T W_U)_{IO} - (x^T W_U)_S = x^T (u_{IO} - u_{S})$$

            where $W_U$ is unembedding matrix, $x$ is the final residual stream value for a single sequence for the final position,
            $u_{IO}$ is the column of $W_U$ corresponding to the correct token, and $u_S$ is the column for the incorrect token. 
            $u_{IO} - u_{S}$ corresponds to the logit difference direction. We will use this term below.  
""")

st.subheader("Logit Lens")
st.write("""
        In this technique, we look at the residual stream after each layer
        and calculate the logit difference from that.

        At each layer, we calculate the logit difference using the residual stream value at that layer, apply layer norm, multiply it with
        the logit difference direction (as if the model didn't have further layers).
         """)

st.image("logit_lens.png")

st.write("""The plot suggests that model writes the correct answer to the residual stream somewhere in
         layers 7 to 9. Makes sense to watch out for these layers in the rest of our exploration.""")

st.subheader("Layer Attribution")
st.write("""
We will now do logit attribution for output of each layer instead of accumulated residual stream value at that layer.
""")

st.image("layer_attribution.png")

st.write("""
We observe that only the attention layers matter suggesting that the task primarily involves
         moving information around. Further, we see layer 9 attention is improves the task performance
         while attention layers 10 and 11 decrease the performance a bit.
""")

st.subheader("Head Attribution")
st.write("""We can further attribute the logit difference to individual heads by decomposing the 
         the attention layer's output into a sum of outputs of each head.""")

st.image("head_attribution.png")

st.write(
    """
    This is a sparse matrix showing that only a few heads matter. Specifically, heads 9.6 and 9.9 seem
    to contribute a lot positively to the logit difference while heads 10.7 and 11.10 contribute a lot 
    negatively.
    These heads actually correspond to some of the name mover and negative name mover heads as discussed 
    in the paper.
    """
)

st.subheader("Attention Analysis")
st.image("positive_attention_head_pattern.jpg")
st.write(
    """
    Attention patterns can be useful to visualize as they tell us what positions the information moves 
    from and to. Since, we are looking at the logit differences at the final sequence position, we only need
    to look at the attention pattern for the final token (in this case, "to").

    For the final token, there's a significant density on the position corresponding to the Indirect Object 
    "John" suggesting that this head maybe responsible for moving the indirect object to the final position.

    Although, this is speculative at this point because the residual stream at layer 9 at the "John"'s position  
    might not necessarily contain information about the token "John".
    """
)

st.header("Activation Patching")
st.write(
    """
    The attribution techniques above only look at the end of the circuit that directly affect the logits.
    As that is not enough, we will now look at activation patching, where we take activations from one run of the model
    and patch them into another run.

    We form a clean and corrupted input that are as close as possilble except for the key detail of the
    Indirect Object. This lets us control for as many of the shared circuits as possible. Then, by patching
    in activations from one run to another, we will not affect the many shared circuits, isolating the 
    circuit we care about.

    We have already tried to control for shared circuits by taking the logit difference.
    """
)
st.subheader("Causal Tracing (Denoising) vs Resample ablation (Noising)")
st.markdown(
    """
    **Causal Tracing** involves patching in clean activations into a corrupted run. The goal is find
    the activations that are *sufficient* to recover clean performance in the context of the circuits
    we care about. Example: If we can patch a head from "The Eiffel Tower is in" (clean input) to 
    "The Colosseum is in" (corrupted input) and flip the answer from Rome to Paris,
    that seems like strong evidence that that head contained the key information
    about the input being the Eiffel Tower.
    * Causal tracing helps in finding activations sufficient for the task.
    * If there is a circuit calculating A OR B, causal tracing can tell us for both A and B that they are
    sufficient on their own.
    * It is important to keep in mind that in our application of this method are we really causing clean
    performance or just breaking the corrupted performance.
    """
)
st.markdown(
    """
    **Resample ablation** does patching the other way around by replacing activations in clean run with
    corrupted activations. It helps find activations that are necessary for good clean performance in the context
    of the circuits we care about. The point is to break the performance on the clean run.
    * If the model has redundancy, we may see nothing is necessary.
    * If there is a circuit calculating A AND B, it tells us that each of A, B being removed will kill
    the performance.
    * This patching need not be from corrupted activations on some prompt, it could also be zero ablation, mean
    ablation, adding gaussian noise etc.
    * The broken performance is better if it's because we have isolated the necessary circuit components
    and not because of uninteresting reasons (eg: throwing the model off-distribution by zero ablation).
    """
)

st.write(
    """
    The results of denoising are considered stronger because:
    * Showing a set of components is sufficient for a task is a big deal.
    * Increase in loss due to zero ablation need not mean the component was important.

    In our patching experiments, the clean run is on original inputs (eg: "When Mary and John went to the store, John gave a drink to")
    and the corrupted run is on the same inputs with the subject token flipped (eg: "When Mary and John went to the store, Mary gave a drink to")
    """
)

st.subheader("IOI Metric")
st.write(
    """
    We will be denoising. So, it makes sense to choose a metric with a value of zero to mean no change
    from the corrupted performance and a value of one to mean clean performance has been completely recovered.
    We will have the metric be a linear function of the logit difference.

    For example, if we patched in the entire clean prompt, we'd get a value of one. If we don't patch at all,
    we'd get a value of zero.
    """
)

st.code(
    """
    (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
    """
)

st.subheader("Residual Stream Patching")
st.markdown(
    """
    Let's start with patching in the residual stream at the start of each layer (aka resid_pre) and for
    each token position. 
    """
)
st.code(
    """
    act_patch_resid_pre = patching.get_act_patch_resid_pre(
        model = model,
        corrupted_tokens = corrupted_tokens,
        clean_cache = clean_cache,
        patching_metric = ioi_metric
    )
    """
)
st.image("activation_patching_figures/resid_pre.png")
st.markdown(
    """
    * The computation by components sufficient for this task seem to be highly localised to two positions
    in the residual stream: S2 token and final token.
    * It seems around layer 8, the information sufficient to predict IO over S is moved from
    S2 token to final token.
    """
)

st.subheader("Residual Stream Patching per block")
st.markdown(
    """
    We can also try patching in the residual stream just after the attention layer (attn_out) 
    or just after the MLP (mlp_out) along with resid_pre as above.
    """
)
st.image("activation_patching_figures/resid_pre_attn_out_mlp_out.png")
st.markdown(
    """
    * Several attention layers are significant, early layers matter on S2 token and layer layers on END token
    * Attention layers 7 and 8 seem to matter, possibly responsible for moving information from S2 to END.
    * MLP layers don't seem to matter much, with an exception: MLP 0
        - MLP 0 apparently is generally important for GPT-2 small, not specific to IOI
    """
)

st.subheader("Attention Head Patching")
st.markdown(
    """
    We can further zoom in and patch individual attention head outputs.
    """
)
st.image("activation_patching_figures/attn_head_out.png")
st.markdown(
    """
    * Some heads we have seen in logit attribution come up here as relevant: 9.9 and 10.7
    * Other heads also show up as relevant: 3.0, 5.5, 6.9, 7.3, 7.9, 8.6, 8.10
    """
n 

st.subheader("Decomposing Attention Heads")
st.subheader("What we have learned so far")

st.header("Path Patching")

st.header("Final Replication")
