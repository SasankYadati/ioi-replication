import streamlit as st

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

st.header("Logit Attribution")

st.subheader("Direction Logit Attribution")
st.subheader("Logit Lens")
st.subheader("Layer Attribution")
st.subheader("Head Attribution")
st.subheader("Attention Analysis")

st.header("Activation Patching")
st.subheader("Noising vs Denoising")
st.subheader("IOI Metric")
st.subheader("Residual Stream Patching")
st.subheader("Residual Stream Patching per block")
st.subheader("Attention Head Patching")
st.subheader("Decomposing Attention Heads")
st.subheader("What we have learned so far")

st.header("Path Patching")

st.header("Final Replication")
