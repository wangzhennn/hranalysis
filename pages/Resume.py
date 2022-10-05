from pathlib import Path

import streamlit as st
from PIL import Image


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"


# --- GENERAL SETTINGS ---
if "my_input" not in st.session_state:
    st.session_state["my_input"]=""
NAME=st.session_state["my_input"]

if "my_input_1" not in st.session_state:
    st.session_state["my_input_1"]=""
DESCRIPTION=st.session_state["my_input_1"]

if "my_input_2" not in st.session_state:
    st.session_state["my_input_2"]=""
EMAIL=st.session_state["my_input_2"]

if "my_input_3" not in st.session_state:
    st.session_state["my_input_3"]=""
PROJECTS=st.session_state["my_input_3"]


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.title(NAME)

with col2:
    st.title(NAME)
    st.write("📒",DESCRIPTION)
    st.write("📫", EMAIL)

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience（1/2）")
if "my_input_4" not in st.session_state:
    st.session_state["my_input_4"]=""
st.write(st.session_state["my_input_4"])

st.write('\n')
st.subheader("Experience（2/2）")
if "my_input_5" not in st.session_state:
    st.session_state["my_input_5"]=""
st.write(st.session_state["my_input_5"])

# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, Pandas), SQL, VBA
- 📊 Data Visulization: PowerBi, MS Excel, Plotly
- 📚 Modeling: Logistic regression, linear regression, decition trees
- 🗄️ Databases: Postgres, MongoDB, MySQL
"""
)


# --- WORK HISTORY ---
st.write('\n')
st.subheader("Work History")
st.write("---")

# --- JOB 1
st.write("🚧", "**Senior Data Analyst | Ross Industries**")
st.write("02/2020 - Present")
st.write(
    """
- ► Used PowerBI and SQL to redeﬁne and track KPIs surrounding marketing initiatives, and supplied recommendations to boost landing page conversion rate by 38%
- ► Led a team of 4 analysts to brainstorm potential marketing and sales improvements, and implemented A/B tests to generate 15% more client leads
- ► Redesigned data model through iterations that improved predictions by 12%
"""
)

# --- JOB 2
st.write('\n')
st.write("🚧", "**Data Analyst | Liberty Mutual Insurance**")
st.write("01/2018 - 02/2022")
st.write(
    """
- ► Built data models and maps to generate meaningful insights from customer data, boosting successful sales eﬀorts by 12%
- ► Modeled targets likely to renew, and presented analysis to leadership, which led to a YoY revenue increase of $300K
- ► Compiled, studied, and inferred large amounts of data, modeling information to drive auto policy pricing
"""
)

# --- JOB 3
st.write('\n')
st.write("🚧", "**Data Analyst | Chegg**")
st.write("04/2015 - 01/2018")
st.write(
    """
- ► Devised KPIs using SQL across company website in collaboration with cross-functional teams to achieve a 120% jump in organic traﬃc
- ► Analyzed, documented, and reported user survey results to improve customer communication processes by 18%
- ► Collaborated with analyst team to oversee end-to-end process surrounding customers' return data
"""
)


# --- Projects & Accomplishments ---
st.write('\n')
st.subheader("Projects & Accomplishments")
st.write("---")
for project, link in PROJECTS.items():
    st.write(f"[{project}]({link})")
