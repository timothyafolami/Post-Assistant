import streamlit as st
from utils import run_graph
import random

# Title of the app
st.title("Post Refiner App")

# adding session state to each user session
session_id = random.randint(0, 100000)
# adding session_id to session state
if "session_id" not in st.session_state:
    st.session_state.session_id = session_id


# Post input
post = st.text_area("Post Input", "Enter the post you want to improve here...", height=220)

# Instruction input
instruction = st.text_area("Instruction Input", "Enter the possible improvements you want here...", height=120)

# Combine the inputs
combined_input = f"Instruction: {instruction}\n\nPost: {post}"

# Placeholder function to simulate model processing
def refine_post(input_text):
    thread_id = st.session_state.session_id
    thread = {"configurable": {"thread_id": thread_id}}
    # adding try and except block to handle the error
    try:
        output = run_graph(input_text, thread=thread)
    except Exception as e:
        output = f"An error occurred, correct the input and try again (you can just try again)."
    return output

# Button to refine the post
if st.button("Refine Post"):
    # creating an animation while the model is processing
    with st.spinner("Refining the post..."):
        # Call the function to refine the post
        refined_output = refine_post(combined_input)
        
    # Display the output
    st.write(refined_output)

