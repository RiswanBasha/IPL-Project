import streamlit as st
import joblib
import pickle
import os
import requests
import dotenv

# Load environment variables (Groq API key etc.)
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"

# Load model and encoders
clf = joblib.load("EDA/ipl_xgboost_best.joblib")
categorical_cols = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
encoders = {col: pickle.load(open(f'EDA/label_encoder_{col}.pkl', 'rb')) for col in categorical_cols}

TEAM_LOGO_DIR = "team_logos"

TEAMS = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bengaluru', 'Sunrisers Hyderabad'
]

VENUES = [
    'Arun Jaitley Stadium, Delhi',
    'Barabati Stadium, Cuttack',
    'Barsapara Cricket Stadium, Guwahati',
    'Brabourne Stadium, Mumbai',
    'Dr DY Patil Sports Academy, Mumbai',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
    'Eden Gardens, Kolkata',
    'Ekana Cricket Stadium, Lucknow',
    'Feroz Shah Kotla Ground, Delhi',
    'Green Park, Kanpur',
    'Holkar Cricket Stadium, Indore',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala',
    'JSCA International Stadium Complex, Ranchi',
    'MA Chidambaram Stadium, Chennai',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Maharashtra Cricket Association Stadium, Pune',
    'M. Chinnaswamy Stadium, Bengaluru',
    'Narendra Modi Stadium, Ahmedabad',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Rajiv Gandhi International Stadium, Hyderabad',
    'Sardar Patel Stadium (Motera), Ahmedabad',
    'Saurashtra Cricket Association Stadium, Rajkot',
    'Shaheed Veer Narayan Singh International Stadium, Raipur',
    'Sawai Mansingh Stadium, Jaipur',
    'Subrata Roy Sahara Stadium, Pune',
    'Vidarbha Cricket Association Stadium, Nagpur',
    'Nehru Stadium, Indore'
]

def overs_to_float(x):
    o, b = str(x).split('.')
    return int(o) + int(b)/6

def get_team_logo_path(team):
    """Try both .png and .jpg for team logos, case-insensitive."""
    base = os.path.join(TEAM_LOGO_DIR, team)
    for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
        path = base + ext
        if os.path.exists(path):
            return path
    # If not found, try with underscores (handle space in file name)
    base = base.replace(' ', '_')
    for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
        path = base + ext
        if os.path.exists(path):
            return path
    return None


def get_commentary_tone(wickets_left, overs_left, runs_left, required_run_rate, run_rate):
    # You can tune these rules as you wish!
    if runs_left <= 12 and wickets_left > 5 and required_run_rate <= run_rate:
        return "comfortable"
    elif wickets_left <= 2 or (runs_left > 18 and required_run_rate > run_rate):
        return "nervous"
    elif overs_left < 1 or abs(required_run_rate - run_rate) < 0.5:
        return "thrilling"
    else:
        return "balanced"


def generate_groq_commentary(batting_team, bowling_team, current_score, target_runs, rr, crr, wickets_left, overs_left, venue, predicted_winner):
    # Calculate context
    runs_left = target_runs - current_score
    overs_left_float = float(overs_left) if isinstance(overs_left, (int, float)) else float(str(overs_left).replace(",", "."))
    tone = get_commentary_tone(wickets_left, overs_left_float, runs_left, rr, crr)
    
    tone_instructions = {
        "comfortable": "This is a straightforward chase, so keep the commentary calm, confident, maybe even a bit celebratory for the batting team.",
        "nervous": "This is a tough chase, so make the commentary tense and dramatic. Focus on the pressure, mention how difficult it is with so few wickets or so many runs, and question if a miracle can happen.",
        "thrilling": "This match is going down to the wire! Make the commentary suspenseful, highlight how close it is, and how one moment could change everything.",
        "balanced": "This is a fair contest. Keep the commentary lively but neutral, mention both teams still have a chance, and don't overhype."
    }
    mood_instruction = tone_instructions[tone]

    prompt = (
        f"Venue: {venue}. {batting_team} need {target_runs - current_score} runs to win, with {wickets_left} wicket(s) in hand and {overs_left} over(s) left. "
        f"Required run rate is {rr}, current run rate is {crr}. "
        f"{10 - int(wickets_left)} wickets have fallen. "
        f"As a cricket commentator, {mood_instruction} "
        f"Based on these facts, say in simple, realistic English who has the upper hand and why. "
        f"You predict {predicted_winner} are likely to win, but mention if a last twist is possible. "
        "Do not use player names or add any details not in the above facts. "
        "Do not talk about football, soccer, or any other sport. "
        "Write 3 to 4 short, clear sentences as if you are on a TV broadcast. "
        "End with one line asking the viewers to stay tuned."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert IPL cricket commentator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.95,
        "max_tokens": 300,
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=45)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Could not generate commentary at this time. (Groq error)"
    except Exception as e:
        return f"Commentary generation failed: {e}"

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="wide")

import base64

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


GROUND_IMAGE_PATH = os.path.join(TEAM_LOGO_DIR, "ground.jpg")
if os.path.exists(GROUND_IMAGE_PATH):
    base64_str = get_base64_of_image(GROUND_IMAGE_PATH)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_str}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


st.title("🏏 IPL Match Winner Predictor")
st.markdown("#### Fill in the current match situation below:")

# --- Input Form ---
with st.form("ipl_form"):
    c1, c2 = st.columns([1, 1])

    # Dynamic dropdowns for teams
    with c1:
        batting_team = st.selectbox("Batting Team", TEAMS, key="batting_team")
        bowling_team_options = [t for t in TEAMS if t != batting_team]
        bowling_team = st.selectbox("Bowling Team", bowling_team_options, key="bowling_team")
        venue = st.selectbox("Venue", VENUES)

    with c2:
        toss_winner = st.selectbox("Toss Winner", TEAMS)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
        remaining_overs = st.text_input("Overs Left (e.g. 2.3 for 2 overs 3 balls)", value="0.5")
        wickets_left = st.slider("Wickets Left", 1, 10, 2)
        current_score = st.number_input("Current Score", min_value=0, max_value=400, value=172)
        target_runs = st.number_input("Target Score", min_value=1, max_value=500, value=180)
    
    # Compute run rates
    try:
        overs_float = overs_to_float(remaining_overs)
        overs_faced = 20 - overs_float  # Assume T20
        run_rate = round(current_score / overs_faced, 2) if overs_faced > 0 else 0
        runs_left = target_runs - current_score
        required_run_rate = round(runs_left / overs_float, 2) if overs_float > 0 else 0
    except:
        run_rate, required_run_rate = 0, 0

    st.write(f"**Current Run Rate:** {run_rate}  |  **Required Run Rate:** {required_run_rate}")
    
    submitted = st.form_submit_button("Predict Winner")

if submitted:
    # Encode features
    try:
        rt_encoded = [
            encoders['batting_team'].transform([batting_team])[0],
            encoders['bowling_team'].transform([bowling_team])[0],
            encoders['venue'].transform([venue])[0],
            encoders['toss_winner'].transform([toss_winner])[0],
            encoders['toss_decision'].transform([toss_decision])[0],
            overs_to_float(remaining_overs),
            wickets_left,
            run_rate,
            required_run_rate
        ]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()
    
    pred = clf.predict([rt_encoded])[0]
    winner_team = bowling_team if pred == 0 else batting_team
    winner_role = "Bowling Team" if pred == 0 else "Batting Team"
    # Centered winner result and logo only
    st.markdown(
        f"<h2 style='text-align: center; color: #FFD700;'>🏆 Predicted Winner: <span style='color:#0099ff'>{winner_team}</span></h2>", 
        unsafe_allow_html=True
    )
    logo_path = get_team_logo_path(winner_team)
    if logo_path:
        # Center the logo and remove caption
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center;">
                <img src="data:image/png;base64,{get_base64_of_image(logo_path)}" style="max-width:320px; width: 40vw;"/>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.info("Logo not found!")


    st.balloons()

    # --- LLM commentary ---
    with st.spinner("Generating thrilling IPL commentary..."):
        commentary = generate_groq_commentary(
            batting_team, bowling_team, current_score, target_runs, 
            required_run_rate, run_rate, wickets_left, remaining_overs, 
            venue, winner_team
        )

    st.markdown("---")
    st.subheader("🎙️ Commentary:")

    # Center and style commentary in a wide column
    centered = st.columns([1, 4, 1])[1]  # Middle column is wider

    with centered:
        st.markdown(
            f"""
            <div style="
                background: #222831;
                color: #F5C518;
                border-radius: 20px;
                padding: 30px 30px 30px 30px;
                margin-top: 16px;
                margin-bottom: 16px;
                font-size: 1.2rem;
                font-weight: bold;
                box-shadow: 0 4px 32px rgba(0,0,0,0.35);
                text-align: center;
                line-height: 2;
            ">
            {commentary}
            </div>
            """,
            unsafe_allow_html=True
        )
