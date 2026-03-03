import streamlit as st
import numpy as np
import random

st.set_page_config(page_title="RL Shopping Assistant", layout="centered")

# -----------------------------
# PLATFORM ACTIONS
# -----------------------------

platforms = ["Amazon", "Flipkart", "AJIO"]

# -----------------------------
# SIMULATED MARKETPLACE DATA
# -----------------------------

marketplace = {
    "tshirt": {
        "Amazon": {"price": 799, "rating": 4.2},
        "Flipkart": {"price": 749, "rating": 4.0},
        "AJIO": {"price": 699, "rating": 4.3},
    },
    "shoes": {
        "Amazon": {"price": 2499, "rating": 4.3},
        "Flipkart": {"price": 2399, "rating": 4.1},
        "AJIO": {"price": 2599, "rating": 4.4},
    },
    "watch": {
        "Amazon": {"price": 1999, "rating": 4.4},
        "Flipkart": {"price": 1899, "rating": 4.1},
        "AJIO": {"price": 2099, "rating": 4.2},
    },
    "electronics": {
        "Amazon": {"price": 18999, "rating": 4.5},
        "Flipkart": {"price": 18499, "rating": 4.2},
        "AJIO": None
    }
}

# -----------------------------
# SESSION STATE INIT
# -----------------------------

if "q_table" not in st.session_state:
    st.session_state.q_table = {}

if "epsilon" not in st.session_state:
    st.session_state.epsilon = 1.0

if "suggested_platform" not in st.session_state:
    st.session_state.suggested_platform = None

if "current_product" not in st.session_state:
    st.session_state.current_product = None

# -----------------------------
# RL PARAMETERS
# -----------------------------

learning_rate = 0.1
discount_factor = 0.9
epsilon_decay = 0.95
min_epsilon = 0.05

# -----------------------------
# INITIALIZE PRODUCT STATE
# -----------------------------

def initialize_product(product):
    if product not in st.session_state.q_table:
        st.session_state.q_table[product] = np.zeros(len(platforms))

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------

def recommend(product):

    initialize_product(product)

    available = marketplace[product]
    valid_platforms = [p for p in platforms if available.get(p) is not None]

    if random.uniform(0, 1) < st.session_state.epsilon:
        chosen = random.choice(valid_platforms)
        st.write("🔎 Exploring...")
    else:
        best_index = np.argmax(st.session_state.q_table[product])
        chosen = platforms[best_index]
        st.write("🎯 Exploiting learned preference...")

    st.session_state.suggested_platform = chosen
    return chosen

# -----------------------------
# UPDATE Q FUNCTION
# -----------------------------

def update_q(product, suggested, reward):

    index = platforms.index(suggested)

    old_value = st.session_state.q_table[product][index]
    next_max = np.max(st.session_state.q_table[product])

    new_value = old_value + learning_rate * (
        reward + discount_factor * next_max - old_value
    )

    st.session_state.q_table[product][index] = new_value

    if st.session_state.epsilon > min_epsilon:
        st.session_state.epsilon *= epsilon_decay


# -----------------------------
# UI START
# -----------------------------

st.title("🛒 RL Smart Shopping Assistant")
st.caption("Learns which platform you prefer for each product")

product = st.text_input("Enter product (tshirt / shoes / watch / electronics):")

if product:
    product = product.lower()

    if product not in marketplace:
        st.error("Product not available in demo dataset.")
    else:
        st.session_state.current_product = product

        st.markdown("### 🏷 Available Platforms")

        for platform, details in marketplace[product].items():
            if details:
                st.write(f"**{platform}** - ₹{details['price']} | {details['rating']}⭐")
            else:
                st.write(f"**{platform}** - Not Available")

        if st.button("🔍 Get Smart Suggestion"):
            suggestion = recommend(product)
            st.success(f"Suggested Platform: {suggestion}")

# -----------------------------
# USER CHOICE
# -----------------------------

if st.session_state.suggested_platform:

    choice = st.selectbox(
        "Where do you want to buy from?",
        [p for p in platforms if marketplace[st.session_state.current_product].get(p) is not None]
    )

    if st.button("Confirm Choice"):

        if choice == st.session_state.suggested_platform:
            reward = 20
            st.success("You accepted suggestion! Positive reward.")
        else:
            reward = -5
            st.warning("You ignored suggestion. Negative reward.")

        update_q(st.session_state.current_product,
                 st.session_state.suggested_platform,
                 reward)

        st.session_state.suggested_platform = None

# -----------------------------
# RESET BUTTON
# -----------------------------

st.markdown("---")

if st.button("🔄 Reset Learning"):
    st.session_state.q_table = {}
    st.session_state.epsilon = 1.0
    st.success("Model Reset!")

# -----------------------------
# DEBUG PANEL
# -----------------------------

with st.expander("📊 View Q-Table"):
    st.write(st.session_state.q_table)
    st.write("Epsilon:", round(st.session_state.epsilon, 3))
