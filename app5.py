#mid term
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_grade_boundaries(df, grade_labels, grade_centric):
    total_students = len(df)
    center_idx = grade_labels.index(grade_centric)
    distribution = np.array([abs(i - center_idx) for i in range(len(grade_labels))])
    distribution = np.max(distribution) - distribution + 1
    distribution = distribution / distribution.sum()
    
    grade_counts = (distribution * total_students).astype(int)
    grade_counts[-1] += total_students - grade_counts.sum()
    
    df_sorted = df.sort_values(by='marks', ascending=False).reset_index(drop=True)
    grade_boundaries = {}
    start_idx = 0
    
    for i, grade in enumerate(grade_labels):
        end_idx = start_idx + grade_counts[i]
        if end_idx > total_students:
            end_idx = total_students
        if start_idx < total_students:
            grade_boundaries[grade] = df_sorted.iloc[end_idx - 1]['marks']
        start_idx = end_idx
    
    return grade_boundaries

def assign_grades(df, grade_boundaries):
    df['grade'] = df['marks'].apply(lambda x: next((g for g, b in grade_boundaries.items() if x >= b), 'F'))
    return df

def plot_grade_distribution(df, grade_labels):
    plt.figure(figsize=(10, 5))
    df['grade'].value_counts().reindex(grade_labels, fill_value=0).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Frequency')
    plt.title('Updated Grade Distribution')
    st.pyplot(plt)

st.title("Advanced Grade Moderation System")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

grade_labels = ['A+', 'A', 'B', 'C', 'D', 'E', 'F']

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    if 'marks' not in df.columns:
        st.error("The uploaded file must contain a 'marks' column.")
    else:
        st.write("Preview of uploaded data:")
        st.write(df.head())
        
        grade_centric = st.selectbox("Select the grade to be most frequent (centered in the bell curve):", grade_labels)
        grade_boundaries = compute_grade_boundaries(df, grade_labels, grade_centric)
        
        st.write("### Adjust Grade Boundaries")
        user_boundaries = {}
        
        for grade in grade_labels:
            user_boundaries[grade] = st.number_input(f"{grade} Cutoff", value=grade_boundaries.get(grade, 0), step=1)
        
        user_boundaries = dict(sorted(user_boundaries.items(), key=lambda x: -x[1]))
        df = assign_grades(df, user_boundaries)
        
        st.write("### Computed Grade Ranges")
        for grade, boundary in user_boundaries.items():
            st.write(f"{grade}: {boundary} and above")
        
        st.write("### Updated Grade Distribution")
        plot_grade_distribution(df, grade_labels)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Updated File", csv, "graded_students.csv", "text/csv")
