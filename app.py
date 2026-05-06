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
epsilon = 0.2

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "q_table" not in st.session_state:
    st.session_state.q_table = {}

if "user_pref" not in st.session_state:
    st.session_state.user_pref = {p: 0 for p in platforms}

if "reward_history" not in st.session_state:
    st.session_state.reward_history = []

# -------------------------------
# INITIALIZE Q-TABLE
# -------------------------------
def initialize_product(product):
    if product not in st.session_state.q_table:
        st.session_state.q_table[product] = np.zeros(len(platforms))

# -------------------------------
# VALID PLATFORMS
# -------------------------------
def get_valid_platforms(product):
    return [p for p in platforms if marketplace[product].get(p) is not None]

# -------------------------------
# RL ACTION SELECTION
# -------------------------------
def choose_action(product):
    initialize_product(product)

    valid_platforms = get_valid_platforms(product)
    valid_indices = [platforms.index(p) for p in valid_platforms]

    q_values = st.session_state.q_table[product]

    if np.random.rand() < epsilon:
        action_index = np.random.choice(valid_indices)
    else:
        valid_q = [(i, q_values[i]) for i in valid_indices]
        action_index = max(valid_q, key=lambda x: x[1])[0]

    return platforms[action_index]

# -------------------------------
# REWARD FUNCTION
# -------------------------------
def calculate_reward(product, platform):

    data = marketplace[product][platform]

    price = data["price"]
    rating = data["rating"]

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

    # RL suggestion
    suggested = choose_action(product)
    st.success(f"🤖 RL Suggestion: {suggested}")

    valid_platforms = get_valid_platforms(product)
    choice = st.selectbox("Where do you want to buy from?", valid_platforms)

    if st.button("Confirm Purchase"):

        reward = calculate_reward(product, choice)

        if choice == suggested:
            reward += 10
            st.success("You followed the RL suggestion 🎯")
        else:
            reward -= 5
            st.warning("You ignored the suggestion ⚠️")

        st.session_state.user_pref[choice] += 1

        update_q(product, choice, reward)

        # store reward history
        st.session_state.reward_history.append(reward)

        st.subheader("📊 Updated Q-Table")
        st.write(st.session_state.q_table[product])

        st.subheader("👤 User Preferences")
        st.write(st.session_state.user_pref)

    # -------------------------------
    # 📈 LEARNING GRAPH
    # -------------------------------
    if len(st.session_state.reward_history) > 1:

        st.subheader("📈 Learning Progress Over Time")

        rewards = st.session_state.reward_history

        df_rewards = pd.DataFrame({
            "Step": list(range(1, len(rewards) + 1)),
            "Reward": rewards
        })

        # Moving average (smooth curve)
        df_rewards["Moving Avg (5)"] = df_rewards["Reward"].rolling(window=5).mean()

        st.line_chart(df_rewards.set_index("Step"))

else:
    if product != "":
        st.error("Product not available in marketplace.")
