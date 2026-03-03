import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RL Smart Shopping Assistant", layout="centered")

st.title("🛍️ RL Smart Shopping Assistant")
st.write("Find the best deal. Let the agent learn your preferences.")

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

# Initialize Q-table in session
if "q_table" not in st.session_state:
    st.session_state.q_table = {}

# -------------------------------
# INITIALIZE PRODUCT STATE
# -------------------------------
def initialize_product(product):
    if product not in st.session_state.q_table:
        st.session_state.q_table[product] = np.zeros(len(platforms))

# -------------------------------
# RECOMMEND CHEAPEST PLATFORM
# -------------------------------
def recommend(product):

    initialize_product(product)

    available = marketplace[product]

    valid_platforms = {
        p: details for p, details in available.items() if details is not None
    }

    cheapest = min(
        valid_platforms,
        key=lambda x: valid_platforms[x]["price"]
    )

    return cheapest

# -------------------------------
# UPDATE Q TABLE
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
# USER INPUT PRODUCT
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
    # SMART SUGGESTION
    # -------------------------------
    suggested = recommend(product)

    st.success(f"💰 Smart Suggestion (Cheapest): {suggested}")

    # -------------------------------
    # USER CHOICE
    # -------------------------------
    choice = st.selectbox("Where do you want to buy from?", platforms)

    if st.button("Confirm Purchase"):

        if marketplace[product][suggested] is not None:

            price = marketplace[product][suggested]["price"]

            reward = 10000 / price   # cheaper → higher reward

            if choice == suggested:
                reward += 10
                st.success("You accepted the smart suggestion! Bonus reward applied.")
            else:
                reward -= 5
                st.warning("You ignored the suggestion. Penalty applied.")

            update_q(product, suggested, reward)

            st.subheader("Updated Q-Table")
            st.write(st.session_state.q_table[product])

else:
    if product != "":
        st.error("Product not available in marketplace.")
