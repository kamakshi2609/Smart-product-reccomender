import streamlit as st
import numpy as np
import random

st.set_page_config(page_title="RL Shopping Assistant", layout="centered")

# -----------------------------
# STATES & ACTIONS
# -----------------------------

states = ["Shoes", "Electronics", "Clothing"]
actions = ["Amazon", "Flipkart", "AJIO", "Cheaper Alternative"]

# Simulated product data
product_data = {
    "Shoes": {
        "Amazon": {"price": 2499, "rating": 4.3},
        "Flipkart": {"price": 2399, "rating": 4.1},
        "AJIO": {"price": 2599, "rating": 4.4},
        "Cheaper Alternative": {"price": 1999, "rating": 3.9},
    },
    "Electronics": {
        "Amazon": {"price": 18999, "rating": 4.5},
        "Flipkart": {"price": 18499, "rating": 4.2},
        "AJIO": {"price": 0, "rating": 0},
        "Cheaper Alternative": {"price": 15999, "rating": 3.8},
    },
    "Clothing": {
        "Amazon": {"price": 1499, "rating": 4.0},
        "Flipkart": {"price": 1399, "rating": 4.1},
        "AJIO": {"price": 1299, "rating": 4.3},
        "Cheaper Alternative": {"price": 999, "rating": 3.7},
    },
}

# -----------------------------
# SESSION STATE INIT
# -----------------------------

if "q_table" not in st.session_state:
    st.session_state.q_table = np.zeros((len(states), len(actions)))

if "epsilon" not in st.session_state:
    st.session_state.epsilon = 1.0

if "current_state" not in st.session_state:
    st.session_state.current_state = None

if "action_index" not in st.session_state:
    st.session_state.action_index = None

# -----------------------------
# Q LEARNING UPDATE
# -----------------------------

def update_q(reward):

    s = st.session_state.current_state
    a = st.session_state.action_index

    lr = 0.1
    gamma = 0.9

    old_value = st.session_state.q_table[s, a]
    next_max = np.max(st.session_state.q_table[s])

    new_value = old_value + lr * (reward + gamma * next_max - old_value)
    st.session_state.q_table[s, a] = new_value

    # Decay epsilon
    if st.session_state.epsilon > 0.05:
        st.session_state.epsilon *= 0.97


# -----------------------------
# UI
# -----------------------------

st.title("🛒 RL Smart Shopping Assistant")
st.caption("Learns which platform you prefer for different product categories")

st.subheader("Select Product Category")

col1, col2, col3 = st.columns(3)

if col1.button("👟 Shoes"):
    st.session_state.current_state = 0

if col2.button("📱 Electronics"):
    st.session_state.current_state = 1

if col3.button("👕 Clothing"):
    st.session_state.current_state = 2


# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------

if st.session_state.current_state is not None:

    state_index = st.session_state.current_state
    category = states[state_index]

    # Epsilon-greedy
    if random.uniform(0, 1) < st.session_state.epsilon:
        action_index = random.randint(0, len(actions) - 1)
    else:
        action_index = np.argmax(st.session_state.q_table[state_index])

    st.session_state.action_index = action_index
    chosen_action = actions[action_index]

    st.markdown("### 🔍 Suggested Platform")

    details = product_data[category][chosen_action]

    if details["price"] == 0:
        st.warning(f"{chosen_action} does not list this category.")
    else:
        st.info(
            f"Platform: {chosen_action}\n\n"
            f"Price: ₹{details['price']}\n\n"
            f"Rating: {details['rating']} ⭐"
        )

    # -----------------------------
    # FEEDBACK
    # -----------------------------

    st.markdown("### 🧠 Your Action")

    col1, col2 = st.columns(2)

    if col1.button("🛍 Clicked / Purchased"):
        update_q(15)
        st.success("Model learned you liked this platform!")

    if col2.button("❌ Ignored"):
        update_q(-5)
        st.error("Model learned this was not preferred.")


# -----------------------------
# RESET
# -----------------------------

st.markdown("---")

if st.button("🔄 Reset Learning"):
    st.session_state.q_table = np.zeros((len(states), len(actions)))
    st.session_state.epsilon = 1.0
    st.success("Model Reset!")


# -----------------------------
# DEBUG PANEL
# -----------------------------

with st.expander("📊 View Q-Table"):
    st.dataframe(st.session_state.q_table)
    st.write("Epsilon:", round(st.session_state.epsilon, 3))
