import streamlit as st

st.title("💡 Simple Streamlit Test App")

# Text input
user_input = st.text_input("Enter some text:")

# Button to display text
if st.button("Submit"):
    if user_input.strip():
        st.write(f"You entered: {user_input}")
    else:
        st.warning("Please type something!")

# Slider example
number = st.slider("Pick a number", 0, 100, 50)
st.write(f"You picked: {number}")

# Checkbox example
if st.checkbox("Show greeting"):
    st.write("Hello! Welcome to Streamlit.")
