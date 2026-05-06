import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RL Smart Shopping Assistant", layout="centered")

st.title("🛍️ RL Smart Shopping Assistant")
st.write("AI-powered deal optimization using Reinforcement Learning")

# -------------------------------
# PLATFORM DATA
# -------------------------------
platforms = ["Amazon", "Flipkart", "AJIO"]

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

# -------------------------------
# RL PARAMETERS
# -------------------------------
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2  # exploration rate

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "q_table" not in st.session_state:
    st.session_state.q_table = {}

if "user_pref" not in st.session_state:
    st.session_state.user_pref = {p: 0 for p in platforms}

# -------------------------------
# INITIALIZE Q-TABLE
# -------------------------------
def initialize_product(product):
    if product not in st.session_state.q_table:
        st.session_state.q_table[product] = np.zeros(len(platforms))

# -------------------------------
# GET VALID PLATFORMS
# -------------------------------
def get_valid_platforms(product):
    available = marketplace[product]
    return [p for p in platforms if available.get(p) is not None]

# -------------------------------
# RL ACTION SELECTION (CORE FIX)
# -------------------------------
def choose_action(product):

    initialize_product(product)

    valid_platforms = get_valid_platforms(product)
    valid_indices = [platforms.index(p) for p in valid_platforms]

    q_values = st.session_state.q_table[product]

    # Exploration vs Exploitation
    if np.random.rand() < epsilon:
        action_index = np.random.choice(valid_indices)
    else:
        valid_q = [(i, q_values[i]) for i in valid_indices]
        action_index = max(valid_q, key=lambda x: x[1])[0]

    return platforms[action_index]

# -------------------------------
# REWARD FUNCTION (UPGRADED)
# -------------------------------
def calculate_reward(product, platform):

    data = marketplace[product][platform]

    price = data["price"]
    rating = data["rating"]

    # Normalize components
    price_score = 10000 / price
    rating_score = rating * 10
    preference_score = st.session_state.user_pref[platform]

    reward = (
        0.6 * price_score +
        0.3 * rating_score +
        0.1 * preference_score
    )

    return reward

# -------------------------------
# Q-TABLE UPDATE
# -------------------------------
def update_q(product, platform, reward):

    index = platforms.index(platform)

    old_value = st.session_state.q_table[product][index]
    next_max = np.max(st.session_state.q_table[product])

    new_value = old_value + learning_rate * (
        reward + discount_factor * next_max - old_value
    )

    st.session_state.q_table[product][index] = new_value

# -------------------------------
# USER INPUT
# -------------------------------
product = st.text_input("🔎 Enter product (tshirt / shoes / watch / electronics)").lower()

if product in marketplace:

    st.subheader("Available Platforms")

    data_rows = []

    for platform, details in marketplace[product].items():
        if details:
            data_rows.append({
                "Platform": platform,
                "Price": details["price"],
                "Rating": details["rating"]
            })

    df = pd.DataFrame(data_rows)
    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # RL-BASED SUGGESTION
    # -------------------------------
    suggested = choose_action(product)

    st.success(f"🤖 RL Suggestion: {suggested}")

    # -------------------------------
    # USER CHOICE
    # -------------------------------
    valid_platforms = get_valid_platforms(product)

    choice = st.selectbox("Where do you want to buy from?", valid_platforms)

    if st.button("Confirm Purchase"):

        reward = calculate_reward(product, choice)

        # Bonus/Penalty
        if choice == suggested:
            reward += 10
            st.success("You followed the RL suggestion 🎯")
        else:
            reward -= 5
            st.warning("You ignored the suggestion ⚠️")

        # Update preference
        st.session_state.user_pref[choice] += 1

        # Update Q-table
        update_q(product, choice, reward)

        st.subheader("📊 Updated Q-Table")
        st.write(st.session_state.q_table[product])

        st.subheader("👤 User Preferences")
        st.write(st.session_state.user_pref)

else:
    if product != "":
        st.error("Product not available in marketplace.")
