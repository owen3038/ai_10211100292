import streamlit as st
import regression
import clustering
import neural_networks
import llm_rag

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Function to change page
def go_to(page_name):
    st.session_state.page = page_name

# Global Modern Style
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f8f9fc;
        }

        .title, .auth-title, .login-title, .welcome-title, .service-title {
            font-size: 48px;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(to right, #6A11CB, #2575FC);
            -webkit-background-clip: text;
            color: transparent;
            margin-top: 30px;
            margin-bottom: 20px;
        }

        .description, .auth-description, .login-description {
            text-align: center;
            font-size: 20px;
            color: #444444;
            margin-bottom: 30px;
        }

        .stButton > button {
            display: block;
            margin: 10px auto;
            padding: 0.9em 2em;
            font-size: 18px;
            font-weight: 600;
            background: linear-gradient(to right, #6A11CB, #2575FC);
            color: white;
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.1);
            transition: 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background: linear-gradient(to right, #5a0fba, #1c63e5);
            transform: translateY(-2px);
        }

        .explore-btn > button, .home-btn > button {
            background: linear-gradient(to right, #6A11CB, #2575FC);
            color: white;
            padding: 0.9em 2.2em;
            border-radius: 10px;
            font-size: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .explore-btn > button:hover, .home-btn > button:hover {
            background: linear-gradient(to right, #5a0fba, #1c63e5);
        }
    </style>
""", unsafe_allow_html=True)

# Home Page
if st.session_state.page == "home":
    st.markdown('<div class="title">🎓 The AI Lab </div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Uncover the power of machine learning and AI through sleek, smart tools built to inspire and educate.</div>', unsafe_allow_html=True)

    if st.button("Let's Get Started"):
        go_to("auth")
        st.rerun()

# Auth Choice Page
elif st.session_state.page == "auth":
    st.markdown('<div class="auth-title">Welcome to AI Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-description">Select one of the following options to continue.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In"):
            go_to("signin")
            st.rerun()
    with col2:
        if st.button("Sign Up"):
            go_to("signup")
            st.rerun()

# Sign Up Page
elif st.session_state.page == "signup":
    st.markdown('<div class="auth-title">Create Your Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-description">Be a part of AI Lab and begin your adventure in AI!</div>', unsafe_allow_html=True)

    with st.form("signup_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")

    if submit:
        if not full_name or not email or not username or not password or not confirm_password:
            st.error("Please fill in all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            st.session_state.user_name = full_name.split()[0]
            go_to("welcome")
            st.rerun()

    # Back Button for Sign Up Page
    if st.button("⬅️ Back to Authentication"):
        go_to("auth")
        st.rerun()

# Sign In Page
elif st.session_state.page == "signin":
    st.markdown('<div class="login-title">Sign In</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-description">Welcome back! Please log in to continue.</div>', unsafe_allow_html=True)

    with st.form("signin_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")

    if submit:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            st.session_state.user_name = username
            go_to("welcome")
            st.rerun()

    # Back Button for Sign In Page
    if st.button("⬅️ Back to Authentication"):
        go_to("auth")
        st.rerun()

# Welcome Page
elif st.session_state.page == "welcome":
    user_name = st.session_state.get("user_name", "User")

    st.markdown(f'<div class="welcome-title">👋 Welcome, {user_name}!</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Ready to explore the power of AI?</div>', unsafe_allow_html=True)

    if st.button("Explore AI Services"):
        go_to("services")
        st.rerun()

# Services Page
elif st.session_state.page == "services":
    st.markdown('<div class="service-title">🧠 AI Services</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('✅ Regression'):
            go_to("regression")
            st.rerun()
    with col2:
        if st.button('🔄 Clustering'):
            go_to("clustering")
            st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        if st.button('🧠 Neural Networks'):
            go_to("neural_networks")
            st.rerun()
    with col4:
        if st.button('💬 Large Language Model'):
            go_to("large_language_model")
            st.rerun()

    if st.button("⬅️ Back to Home"):
        go_to("home")
        st.rerun()

# Service Pages Routing
elif st.session_state.page == "regression":
    regression.run()
elif st.session_state.page == "clustering":
    clustering.run()
elif st.session_state.page == "neural_networks":
    neural_networks.run()
elif st.session_state.page == "large_language_model":
    llm_rag.run()
