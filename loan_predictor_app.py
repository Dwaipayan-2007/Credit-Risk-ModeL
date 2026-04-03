import streamlit as st
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered",
)

# Load model

@st.cache_resource
def load_model():
    return joblib.load("credit model.pkl")

model = load_model()

# Features

def get_age_group(age):
    if age <= 25:   return 0
    elif age <= 40: return 1
    elif age <= 60: return 2
    else:           return 3

def get_job_stability(job):
    if job in {"Retired", "Management"}:  return 2
    if job in {"Student", "Unemployed"}:  return 0
    return 1

def build_features(age, job, marital, education, housing_loan, personal_loan, last_outcome):
    job_map = {
        "Admin": "job_admin.", "Blue-Collar": "job_blue-collar",
        "Entrepreneur": "job_entrepreneur", "Housemaid": "job_housemaid",
        "Retired": "job_retired", "Self-Employed": "job_self-employed",
        "Services": "job_services", "Student": "job_student",
        "Technician": "job_technician",
    }
    job_feats = {k: 0 for k in ["job_admin.", "job_blue-collar", "job_entrepreneur",
                                "job_housemaid", "job_retired", "job_self-employed",
                                "job_services", "job_student", "job_technician"]}
    col = job_map.get(job)
    if col:
        job_feats[col] = 1

    has_degree = 1 if education in ("Professional Course", "University Degree") else 0
    outcome_success = 1 if last_outcome == "Success" else 0

    return {
        **job_feats,
        "marital_divorced":1 if marital == "Divorced" else 0,
        "education_high.school":1 if education == "High School" else 0,
        "poutcome_failure":1 if last_outcome == "Failure" else 0,
        "basic_education":1 if education in ("Basic 4y", "Basic 6y") else 0,
        "Financial_Burden":(1 if housing_loan else 0) + (1 if personal_loan else 0),
        "Age_Group":get_age_group(age),
        "Prof_Maturity":age * has_degree,
        "Age_Success_Index":age * outcome_success,
        "Job_Stability":get_job_stability(job),
        "Social_Stability":has_degree + (1 if marital == "Married" else 0),
    }

order = [
    "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid",
    "job_retired", "job_self-employed", "job_services", "job_student",
    "job_technician", "marital_divorced", "education_high.school",
    "poutcome_failure", "basic_education", "Financial_Burden",
    "Age_Group", "Prof_Maturity", "Age_Success_Index",
    "Job_Stability", "Social_Stability",
]

# HEADER

st.title("🏦 Loan Approval Checker")
st.write("Answer a few simple questions and the AI model will predict whether a loan application is likely to be **approved** or **denied**.")

# STEP 1 — PERSONAL INFO

st.subheader("Step 1 of 4 — 👤 About the Applicant")
st.caption("Basic personal details — helps the model understand their life stage and responsibilities.")

age = st.slider(
    "How old is the applicant?",
    min_value=18, max_value=69, value=35, step=1,
)

age_group_labels = {
    0: " Young Adult (18 - 25)",
    1: " Adult (26 - 40)",
    2: " Mid Aged (41 - 60)",
    3: " Senior Citizen (61 - 69)",
}
st.caption(f" Age group: {age_group_labels[get_age_group(age)]}")

marital = st.selectbox(
    "What is their marital status?",
    ["Married", "Single", "Divorced", "Unknown"]
)

education = st.selectbox(
    "What is their highest level of education?",
    ["University Degree", "Professional Course", "High School",
     "Basic 9y", "Basic 6y", "Basic 4y", "Illiterate", "Unknown"]
)
st.divider()

# STEP 2 — EMPLOYMENT

st.subheader("Step 2 of 4 — 💼 Employment")
st.caption("Job type tells the model about income reliability.")

job = st.selectbox(
    "What type of job does the applicant have?",
    ["Admin", "Blue-Collar", "Entrepreneur", "Housemaid",
     "Management", "Retired", "Self-Employed", "Services",
     "Student", "Technician", "Unemployed", "Unknown"]
)

stability_map = {0: "🔴 Low", 1: "🟡 Moderate", 2: "🟢 High"}
stab = get_job_stability(job)
st.caption(f"Income stability: **{stability_map[stab]}**")
st.divider()

# STEP 3 — FINANCIAL SITUATION

st.subheader("Step 3 of 4 — 💰 Current Financial Situation")
st.caption("Existing loans mean the applicant already has repayment commitments.")

col1, col2 = st.columns(2)
with col1:
    housing_loan = st.radio(
        "Do they have a housing / mortgage loan?",
        ["No", "Yes"],
        horizontal=True,
    ) == "Yes"
with col2:
    personal_loan = st.radio(
        "Do they have a personal loan?",
        ["No", "Yes"],
        horizontal=True,
    ) == "Yes"

burden = (1 if housing_loan else 0) + (1 if personal_loan else 0)
if burden == 0:
    st.write("✅ No existing loans — lowest financial burden. Good sign!")
elif burden == 1:
    st.write("🟡 One active loan — moderate financial burden.")
else:
    st.write("🔴 Both loans active — high financial burden.")
st.divider()

# STEP 4 — PREVIOUS BANK CONTACT

st.subheader("Step 4 of 4 — 📞 Previous Bank Contact")
st.caption("Has the bank contacted this person before?")

last_outcome_display = st.selectbox(
    "What happened the last time the bank contacted this person?",
    [
        "They have never been contacted before",
        "The previous contact went well (they said yes) ✅",
        "The previous contact did not work out ❌",
    ]
)

outcome_map = {
    "They have never been contacted before":"Nonexistent",
    "The previous contact went well (they said yes) ✅":"Success",
    "The previous contact did not work out ❌":"Failure",
}
last_outcome_val = outcome_map[last_outcome_display]
st.divider()

# PREDICT BUTTON

predict_btn = st.button("🔍  Check Loan Approval", use_container_width=True)

# RESULT

if predict_btn:
    feat = build_features(age, job, marital, education,
                          housing_loan, personal_loan, last_outcome_val)
    X = pd.DataFrame([feat])[order]
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    prob_approve = proba[1] * 100
    prob_deny = proba[0] * 100
    st.divider()

    # Main verdict

    st.header("📋 Prediction Result")

    if prediction == 1:
        st.success(f"✅ Loan Likely APPROVED")
    else:
        st.error(f"❌ Loan Likely DENIED")

    st.subheader("📊 Approval vs Denial Probability")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("✅ Approval Chance", f"{prob_approve:.1f}%")
        st.progress(int(prob_approve))
    with col_b:
        st.metric("❌ Denial Chance", f"{prob_deny:.1f}%")
        st.progress(int(prob_deny))
    st.divider()

    # Key factors

    st.subheader("🔍 Key Factors That Influenced This Result")

    positive_factors, negative_factors, neutral_factors = [], [], []

    if last_outcome_val == "Success":
        positive_factors.append("✅ Successful past bank contact")
    elif last_outcome_val == "Failure":
        negative_factors.append("❌ Failed past bank contact")
    else:
        neutral_factors.append("🚫 No previous contact history")

    if education in ("University Degree", "Professional Course"):
        positive_factors.append("🎓 Higher education level")
    elif education in ("Basic 4y", "Basic 6y", "Basic 9y", "Illiterate"):
        negative_factors.append("📚 Lower education level")
    else:
        neutral_factors.append(f"🏫 Education: {education}")

    if stab == 2:
        positive_factors.append(f"🌟 Highly stable job")
    elif stab == 0:
        negative_factors.append(f"⚠️ Low job stability")
    else:
        neutral_factors.append(f"💼 Moderate stability")

    if burden == 0:
        positive_factors.append("🟢 No existing loans")
    elif burden == 1:
        neutral_factors.append("🟡 One existing loan")
    else:
        negative_factors.append("🔴 Two existing loans")

    if marital == "Married":
        positive_factors.append("👫 Married — possible joint household support")
    elif marital == "Divorced":
        neutral_factors.append("📋 Divorced")
    else:
        neutral_factors.append(f"👤 Marital status: {marital}")

    age_lbl = {0: "Young adult (18 - 25)", 1: "Adult (26 - 40)", 2: "Mid-career (41 - 60)", 3: "Senior (61 - 69)"}
    neutral_factors.append(f"🎂 Age {age} — {age_lbl[get_age_group(age)]}")

    if positive_factors:
        st.write("**🟢 Factors that helped the application:**")
        for f in positive_factors:
            st.write(f"- {f}")

    if negative_factors:
        st.write("**🔴 Factors that hurt the application:**")
        for f in negative_factors:
            st.write(f"- {f}")

    if neutral_factors:
        st.write("**⚪ Other information considered:**")
        for f in neutral_factors:
            st.write(f"- {f}")
    st.divider()

else:
    st.write("📝 Complete all 4 steps above, then click **Check Loan Approval** to see the result.")