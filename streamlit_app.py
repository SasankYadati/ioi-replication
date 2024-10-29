import streamlit as st

st.title("Replicating Interpretability in the Wild")
st.caption("We find a circuit in GPT-2 small that performs the Indirect Object Identification (IOI) task.")

st.markdown("[Original paper](https://arxiv.org/pdf/2211.00593)")
st.markdown("[Reference](https://arena3-chapter1-transformer-interp.streamlit.app/[1.4.1]_Indirect_Object_Identification)")

st.header("Setup", divider=True)

st.subheader("Load GPT-2 small")

st.subheader("Performance on IOI task")

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
