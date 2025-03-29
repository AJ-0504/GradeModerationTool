import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --- Helper Functions ---

def compute_grade_boundaries(df, grade_labels, grade_centric, manual_boundaries=None):
    """Compute grade boundaries based on a centric grade or manual overrides."""
    total_students = len(df)
    center_idx = grade_labels.index(grade_centric)
    
    if manual_boundaries:
        grade_boundaries = manual_boundaries
    else:
        # Create a bell-like distribution centered around the chosen grade
        distribution = np.array([abs(i - center_idx) for i in range(len(grade_labels))])
        distribution = np.max(distribution) - distribution + 1
        distribution = distribution / distribution.sum()
        
        grade_counts = (distribution * total_students).astype(int)
        grade_counts[-1] += total_students - grade_counts.sum()  # Adjust last grade to match total
        
        df_sorted = df.sort_values(by='marks', ascending=False).reset_index(drop=True)
        grade_boundaries = {}
        start_idx = 0
        
        for i, grade in enumerate(grade_labels[:-1]):
            end_idx = start_idx + grade_counts[i]
            if end_idx > total_students:
                end_idx = total_students
            if start_idx < total_students:
                grade_boundaries[grade] = df_sorted.iloc[end_idx - 1]['marks']
            start_idx = end_idx
        
        grade_boundaries[grade_labels[-1]] = 0
    
    # Assign grades based on boundaries
    df['grade'] = df['marks'].apply(lambda x: next((g for g, b in grade_boundaries.items() if x >= b), grade_labels[-1]))
    return df, grade_boundaries

def validate_boundaries(grade_labels, boundaries):
    """Ensure grade boundaries are in descending order."""
    boundary_values = [boundaries[grade] for grade in grade_labels[:-1]]
    return all(boundary_values[i] > boundary_values[i+1] for i in range(len(boundary_values) - 1))

def plot_grade_distribution(df, grade_labels):
    """Plot the distribution of grades using Plotly."""
    grade_counts = df['grade'].value_counts().reindex(grade_labels, fill_value=0)
    fig = px.bar(x=grade_counts.index, y=grade_counts.values, 
                 labels={'x': 'Grades', 'y': 'Frequency'},
                 title='Grade Distribution', 
                 color=grade_counts.index, 
                 text=grade_counts.values)
    st.plotly_chart(fig)

def clean_data(df):
    """Clean the dataframe by converting marks to numeric and removing invalid rows."""
    df['marks'] = pd.to_numeric(df['marks'], errors='coerce')
    invalid_rows = df[df['marks'].isna()]
    if not invalid_rows.empty:
        st.warning(f"Found {len(invalid_rows)} rows with invalid marks. These will be removed.")
        df = df.dropna(subset=['marks'])
    return df

# --- Main Application ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="AI-Powered Grading System", layout="wide")
    st.title("📊 AI-Powered Grade Moderation System")

    # Sidebar: Custom Grade Labels
    st.sidebar.write("### 🎓 Custom Grade Labels")
    custom_labels = st.sidebar.text_input("Enter custom grade labels (comma-separated, e.g., 'A,B,C,D,F'):", "")
    if custom_labels:
        grade_labels = [label.strip() for label in custom_labels.split(',')]
    else:
        grade_labels = ['A+', 'A', 'B', 'C', 'D', 'E', 'F']
    
    if len(grade_labels) < 2:
        st.error("🚨 Please provide at least two grade labels.")
        return
    
    grade_centric = st.selectbox("🎯 Select the grade to be most frequent (centered in the bell curve):", grade_labels)
    
    # File Upload
    uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Load and clean data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        
        if 'marks' not in df.columns:
            st.error("🚨 The uploaded file must contain a 'marks' column.")
            return
        
        df = clean_data(df)
        
        # Create tabs for Grading and Statistics & Visualizations
        tab1, tab2 = st.tabs(["Grading", "Statistics & Visualizations"])
        
        # --- Grading Tab ---
        with tab1:
            df, boundaries = compute_grade_boundaries(df, grade_labels, grade_centric)
            
            # Manual Boundary Adjustments
            st.sidebar.write("### ✏️ Adjust Grade Ranges (Manual Override)")
            manual_boundaries = {}
            for grade in grade_labels[:-1]:
                manual_boundaries[grade] = st.sidebar.slider(f"{grade} minimum marks", 
                                                             0.0, 100.0, 
                                                             float(boundaries[grade]), 
                                                             step=0.1)
            manual_boundaries[grade_labels[-1]] = 0
            
            if not validate_boundaries(grade_labels, manual_boundaries):
                st.sidebar.error("🚨 Grade boundaries must be in descending order. Please adjust the sliders.")
            else:
                df, boundaries = compute_grade_boundaries(df, grade_labels, grade_centric, manual_boundaries)
                
                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### 📌 Computed Grade Ranges")
                    boundary_df = pd.DataFrame(list(boundaries.items()), columns=['Grade', 'Minimum Marks'])
                    st.table(boundary_df)
                
                with col2:
                    st.write("### 📊 Updated Grade Distribution")
                    plot_grade_distribution(df, grade_labels)
                
                st.write("### 📝 Updated Data Preview")
                st.dataframe(df.head(20))
                
                # Download Option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Updated File", csv, "graded_students.csv", "text/csv")
        
        # --- Statistics & Visualizations Tab ---
        with tab2:
            st.write("### 📈 Statistics")
            stats = {
                "Mean": df['marks'].mean(),
                "Median": df['marks'].median(),
                "Mode": df['marks'].mode().values[0] if not df['marks'].mode().empty else "N/A",
                "Standard Deviation": df['marks'].std(),
                "Minimum": df['marks'].min(),
                "Maximum": df['marks'].max(),
            }
            stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
            st.table(stats_df)
            
            st.write("### 📊 Visualizations")
            plot_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot", "Bar Chart"])
            
            if plot_type == "Histogram":
                fig = px.histogram(df, x='marks', title='Distribution of Marks', nbins=20)
                st.plotly_chart(fig)
            elif plot_type == "Box Plot":
                fig = px.box(df, y='marks', title='Box Plot of Marks')
                st.plotly_chart(fig)
            elif plot_type == "Bar Chart":
                if 'grade' in df.columns:
                    grade_counts = df['grade'].value_counts().reindex(grade_labels, fill_value=0)
                    fig = px.bar(x=grade_counts.index, y=grade_counts.values, 
                                 labels={'x': 'Grades', 'y': 'Frequency'},
                                 title='Grade Distribution')
                    st.plotly_chart(fig)
                else:
                    st.write("Please assign grades in the 'Grading' tab to see the bar chart.")

# Run the app
if __name__ == "__main__":
    main()
