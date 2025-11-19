import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Tuition Centre Analytics Dashboard",
    page_icon="graph",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Advanced Tuition Centre Analytics Dashboard")
st.markdown("Comprehensive analysis of student performance using advanced statistics")

# Load data function
@st.cache_data
def load_data():
    data = {
        'Student_ID': ['ST001', 'ST002', 'ST003', 'ST004', 'ST005', 'ST006', 'ST007', 'ST008', 'ST009', 'ST010',
                      'ST011', 'ST012', 'ST013', 'ST014', 'ST015', 'ST016', 'ST017', 'ST018', 'ST019', 'ST020',
                      'ST021', 'ST022', 'ST023', 'ST024', 'ST025', 'ST026', 'ST027', 'ST028', 'ST029', 'ST030',
                      'ST031', 'ST032', 'ST033', 'ST034', 'ST035', 'ST036', 'ST037', 'ST038', 'ST039', 'ST040',
                      'ST041', 'ST042', 'ST043', 'ST044', 'ST045', 'ST046', 'ST047', 'ST048', 'ST049', 'ST050'],
        'Student_Name': ['Samuel Mwangi', 'Dennis Kariuki', 'Kevin Kiptoo', 'Vivian Obiero', 'Rose Wanjiku',
                        'Ruth Naliaka', 'John Ochieng', 'John Ochieng', 'Vivian Obiero', 'Kevin Kiptoo',
                        'Ruth Naliaka', 'Ruth Naliaka', 'Faith Achieng', 'Samuel Mwangi', 'Paul Mwende',
                        'Dennis Kariuki', 'Vivian Obiero', 'Kevin Kiptoo', 'Vivian Obiero', 'Faith Achieng',
                        'Vivian Obiero', 'Brian Otieno', 'Mercy Chebet', 'John Ochieng', 'Faith Achieng',
                        'Samuel Mwangi', 'Mercy Chebet', 'Ruth Naliaka', 'Samuel Mwangi', 'Diana Wekesa',
                        'Kevin Kiptoo', 'Mercy Chebet', 'Peter Ouma', 'Samuel Mwangi', 'Ruth Naliaka',
                        'Faith Achieng', 'Diana Wekesa', 'Lilian Anyango', 'Mercy Chebet', 'Diana Wekesa',
                        'John Ochieng', 'Rose Wanjiku', 'Rose Wanjiku', 'Felix Onyango', 'Felix Onyango',
                        'John Ochieng', 'Dennis Kariuki', 'Brian Otieno', 'Paul Mwende', 'Vivian Obiero'],
        'Subject': ['Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
                   'English', 'English', 'Mathematics', 'English', 'English', 'Mathematics', 'English',
                   'Mathematics', 'Mathematics', 'Mathematics', 'English', 'Mathematics', 'English', 'Mathematics',
                   'English', 'Mathematics', 'Mathematics', 'English', 'English', 'English', 'Mathematics',
                   'English', 'English', 'Mathematics', 'Mathematics', 'English', 'English', 'Mathematics',
                   'English', 'Mathematics', 'Mathematics', 'Mathematics', 'English', 'English', 'English',
                   'English', 'Mathematics', 'Mathematics', 'Mathematics', 'English', 'English', 'Mathematics',
                   'English', 'Mathematics'],
        'Topic': ['Statistics', 'Trigonometry', 'Algebra', 'Geometry', 'Trigonometry', 'Statistics', 'Comprehension',
                 'Grammar', 'Algebra', 'Grammar', 'Grammar', 'Geometry', 'Grammar', 'Statistics', 'Algebra',
                 'Trigonometry', 'Composition', 'Statistics', 'Comprehension', 'Statistics', 'Comprehension',
                 'Algebra', 'Statistics', 'Composition', 'Literature', 'Composition', 'Statistics', 'Literature',
                 'Comprehension', 'Statistics', 'Statistics', 'Grammar', 'Grammar', 'Trigonometry', 'Comprehension',
                 'Geometry', 'Statistics', 'Statistics', 'Literature', 'Literature', 'Comprehension', 'Grammar',
                 'Statistics', 'Geometry', 'Trigonometry', 'Comprehension', 'Grammar', 'Statistics', 'Comprehension',
                 'Algebra'],
        'Test_Date': ['2025-09-01', '2025-09-21', '2025-09-09', '2025-09-03', '2025-09-08', '2025-09-11',
                     '2025-09-30', '2025-09-28', '2025-09-15', '2025-09-17', '2025-09-26', '2025-09-30',
                     '2025-09-05', '2025-09-01', '2025-09-12', '2025-09-24', '2025-09-17', '2025-09-21',
                     '2025-09-29', '2025-09-20', '2025-09-12', '2025-09-28', '2025-09-01', '2025-09-06',
                     '2025-09-18', '2025-09-06', '2025-09-18', '2025-09-01', '2025-09-15', '2025-09-14',
                     '2025-09-06', '2025-09-01', '2025-09-29', '2025-09-16', '2025-09-06', '2025-09-12',
                     '2025-09-16', '2025-09-13', '2025-09-11', '2025-09-26', '2025-09-19', '2025-09-24',
                     '2025-09-28', '2025-09-17', '2025-09-08', '2025-09-10', '2025-09-20', '2025-09-05',
                     '2025-09-11', '2025-09-29'],
        'Day': ['Monday', 'Tuesday', 'Tuesday', 'Wednesday', 'Monday', 'Thursday', 'Tuesday', 'Tuesday',
               'Monday', 'Wednesday', 'Friday', 'Tuesday', 'Friday', 'Monday', 'Friday', 'Wednesday',
               'Wednesday', 'Tuesday', 'Monday', 'Monday', 'Friday', 'Tuesday', 'Monday', 'Monday',
               'Thursday', 'Monday', 'Thursday', 'Monday', 'Monday', 'Tuesday', 'Monday', 'Monday',
               'Monday', 'Tuesday', 'Monday', 'Friday', 'Tuesday', 'Monday', 'Thursday', 'Friday',
               'Friday', 'Wednesday', 'Tuesday', 'Wednesday', 'Monday', 'Wednesday', 'Monday', 'Friday',
               'Thursday', 'Monday'],
        'Teacher_Name': ['Mrs. Moraa', 'Mrs. Atieno', 'Mr. Kamau', 'Mr. Njoroge', 'Mr. Njoroge', 'Mr. Kamau',
                        'Mr. Njoroge', 'Mrs. Moraa', 'Mrs. Moraa', 'Mr. Kamau', 'Mr. Kamau', 'Mr. Njoroge',
                        'Mrs. Moraa', 'Mr. Njoroge', 'Mrs. Atieno', 'Mrs. Moraa', 'Mr. Kamau', 'Mr. Njoroge',
                        'Mrs. Moraa', 'Mr. Njoroge', 'Mrs. Atieno', 'Mr. Njoroge', 'Mrs. Atieno', 'Mr. Njoroge',
                        'Mrs. Moraa', 'Mrs. Atieno', 'Mr. Njoroge', 'Mrs. Moraa', 'Mr. Kamau', 'Mrs. Atieno',
                        'Mr. Kamau', 'Mr. Njoroge', 'Mrs. Atieno', 'Mr. Kamau', 'Mrs. Moraa', 'Mr. Njoroge',
                        'Mrs. Atieno', 'Mrs. Moraa', 'Mrs. Moraa', 'Mr. Kamau', 'Mr. Kamau', 'Mr. Njoroge',
                        'Mrs. Atieno', 'Mr. Kamau', 'Mrs. Atieno', 'Mr. Kamau', 'Mrs. Atieno', 'Mrs. Moraa',
                        'Mr. Njoroge', 'Mr. Kamau'],
        'Score': [93, 62, 87, 81, 67, 76, 62, 79, 54, 63, 73, 93, 45, 45, 69, 87, 84, 50, 54, 45,
                 85, 65, 67, 64, 64, 64, 68, 57, 40, 85, 52, 77, 86, 60, 67, 40, 43, 69, 91, 67,
                 72, 65, 51, 51, 78, 52, 74, 66, 55, 87]
    }
    df = pd.DataFrame(data)
    df['Test_Date'] = pd.to_datetime(df['Test_Date'])
    return df

# Load data
df = load_data()

# Sidebar - Filters and Controls
st.sidebar.title("🎛 Dashboard Controls")

# File upload section
st.sidebar.header("Data Import")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df['Test_Date'] = pd.to_datetime(df['Test_Date'])
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Filters
st.sidebar.header("Data Filters")
selected_subject = st.sidebar.selectbox('Subject', ['All'] + list(df['Subject'].unique()))
selected_teacher = st.sidebar.selectbox('Teacher', ['All'] + list(df['Teacher_Name'].unique()))
selected_topic = st.sidebar.selectbox('Topic', ['All'] + list(df['Topic'].unique()))

# Date range filter
min_date = df['Test_Date'].min()
max_date = df['Test_Date'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Apply filters
filtered_df = df.copy()
if selected_subject != 'All':
    filtered_df = filtered_df[filtered_df['Subject'] == selected_subject]
if selected_teacher != 'All':
    filtered_df = filtered_df[filtered_df['Teacher_Name'] == selected_teacher]
if selected_topic != 'All':
    filtered_df = filtered_df[filtered_df['Topic'] == selected_topic]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['Test_Date'] >= pd.to_datetime(date_range[0])) & 
        (filtered_df['Test_Date'] <= pd.to_datetime(date_range[1]))
    ]

# Main Dashboard
st.markdown("---")

# Key Metrics with NumPy calculations
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

scores_array = np.array(filtered_df['Score'])

with col1:
    avg_score = np.mean(scores_array)
    st.metric("Average Score", f"{avg_score:.1f}", delta=f"{avg_score - 50:.1f} vs 50")

with col2:
    median_score = np.median(scores_array)
    st.metric("Median Score", f"{median_score:.1f}")

with col3:
    std_dev = np.std(scores_array)
    st.metric("Standard Deviation", f"{std_dev:.1f}")

with col4:
    pass_rate = np.mean(scores_array >= 50) * 100
    st.metric("Pass Rate", f"{pass_rate:.1f}%")

# Advanced Statistics with SciPy
st.markdown("---")
st.header("🔬 Advanced Statistical Analysis")

col5, col6, col7, col8 = st.columns(4)

with col5:
    skewness = stats.skew(scores_array)
    st.metric("Skewness", f"{skewness:.2f}")

with col6:
    kurtosis = stats.kurtosis(scores_array)
    st.metric("Kurtosis", f"{kurtosis:.2f}")

with col7:
    variance = np.var(scores_array)
    st.metric("Variance", f"{variance:.1f}")

with col8:
    cv = (std_dev / avg_score) * 100
    st.metric("Coefficient of Variation", f"{cv:.1f}%")

# Performance Categories using NumPy
st.subheader("Performance Distribution")
performance_categories = {
    'Excellent (80-100)': np.sum((scores_array >= 80) & (scores_array <= 100)),
    'Good (60-79)': np.sum((scores_array >= 60) & (scores_array < 80)),
    'Average (50-59)': np.sum((scores_array >= 50) & (scores_array < 60)),
    'Needs Improvement (<50)': np.sum(scores_array < 50)
}

fig_pie = px.pie(
    values=list(performance_categories.values()),
    names=list(performance_categories.keys()),
    title="Performance Categories Distribution"
)
st.plotly_chart(fig_pie, use_container_width=True)

# Visualizations
st.markdown("---")
st.header("Interactive Visualizations")

# Row 1: Score Distribution and Teacher Performance
col1, col2 = st.columns(2)

with col1:
    # Histogram with normal distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=filtered_df['Score'], 
        name='Score Distribution',
        nbinsx=20,
        opacity=0.7
    ))
    
    # Add normal distribution curve
    x_norm = np.linspace(scores_array.min(), scores_array.max(), 100)
    y_norm = stats.norm.pdf(x_norm, avg_score, std_dev) * len(scores_array) * (scores_array.max() - scores_array.min()) / 20
    
    fig_hist.add_trace(go.Scatter(
        x=x_norm, y=y_norm, mode='lines', name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    fig_hist.update_layout(title='Score Distribution with Normal Curve')
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Teacher performance with confidence intervals
    teacher_stats = filtered_df.groupby('Teacher_Name').agg({
        'Score': ['mean', 'std', 'count']
    }).round(2)
    teacher_stats.columns = ['Mean', 'Std', 'Count']
    teacher_stats = teacher_stats.reset_index()
    
    # Calculate 95% confidence intervals
    teacher_stats['CI_lower'] = teacher_stats['Mean'] - 1.96 * teacher_stats['Std'] / np.sqrt(teacher_stats['Count'])
    teacher_stats['CI_upper'] = teacher_stats['Mean'] + 1.96 * teacher_stats['Std'] / np.sqrt(teacher_stats['Count'])
    
    fig_teacher = go.Figure()
    fig_teacher.add_trace(go.Bar(
        x=teacher_stats['Teacher_Name'],
        y=teacher_stats['Mean'],
        name='Average Score',
        error_y=dict(
            type='data',
            array=teacher_stats['Mean'] - teacher_stats['CI_lower'],
            visible=True
        )
    ))
    fig_teacher.update_layout(title='Teacher Performance with Confidence Intervals')
    st.plotly_chart(fig_teacher, use_container_width=True)

# Row 2: Topic Analysis and Trend Analysis
col3, col4 = st.columns(2)

with col3:
    # Topic performance heatmap
    topic_subject_avg = filtered_df.groupby(['Topic', 'Subject'])['Score'].mean().unstack(fill_value=0)
    fig_heatmap = px.imshow(
        topic_subject_avg,
        title='Average Scores: Topic vs Subject',
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col4:
    # Time series analysis with moving average
    daily_avg = filtered_df.groupby('Test_Date')['Score'].mean().reset_index()
    if len(daily_avg) > 1:
        window = min(3, len(daily_avg))
        daily_avg['Moving_Avg'] = daily_avg['Score'].rolling(window=window).mean()
        
        fig_trend = px.line(
            daily_avg, 
            x='Test_Date', 
            y=['Score', 'Moving_Avg'],
            title='Performance Trend with Moving Average',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# Student Analytics
st.markdown("---")
st.header("Student Performance Analytics")

# Top and bottom performers
col9, col10 = st.columns(2)

with col9:
    student_avg = filtered_df.groupby(['Student_ID', 'Student_Name'])['Score'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    student_avg.columns = ['Average_Score', 'Std_Dev', 'Test_Count']
    student_avg = student_avg.reset_index()
    
    # Calculate percentiles
    avg_scores = student_avg['Average_Score'].values
    student_avg['Percentile'] = [stats.percentileofscore(avg_scores, score) for score in avg_scores]
    
    top_students = student_avg.nlargest(10, 'Average_Score')
    st.subheader("Top 10 Performers")
    st.dataframe(top_students, use_container_width=True)

with col10:
    # Student progress tracking
    st.subheader("Student Progress Analysis")
    selected_student = st.selectbox('Select Student:', filtered_df['Student_Name'].unique())
    
    if selected_student:
        student_data = filtered_df[filtered_df['Student_Name'] == selected_student].sort_values('Test_Date')
        
        if len(student_data) > 1:
            fig_student = px.line(
                student_data, 
                x='Test_Date', 
                y='Score',
                title=f'{selected_student} - Performance Timeline',
                markers=True
            )
            st.plotly_chart(fig_student, use_container_width=True)

# Statistical Tests
st.markdown("---")
st.header("Statistical Hypothesis Testing")

col11, col12 = st.columns(2)

with col11:
    # T-test between subjects
    math_scores = filtered_df[filtered_df['Subject'] == 'Mathematics']['Score']
    english_scores = filtered_df[filtered_df['Subject'] == 'English']['Score']
    
    if len(math_scores) > 1 and len(english_scores) > 1:
        t_stat, p_value = stats.ttest_ind(math_scores, english_scores, nan_policy='omit')
        
        st.subheader("Subject Comparison T-Test")
        st.write(f"*Mathematics vs English Performance*")
        st.write(f"T-statistic: {t_stat:.3f}")
        st.write(f"P-value: {p_value:.3f}")
        st.write(f"*Interpretation:* {'Significant difference' if p_value < 0.05 else 'No significant difference'}")

with col12:
    # Normality test
    st.subheader("Normality Test (Shapiro-Wilk)")
    if len(scores_array) > 3 and len(scores_array) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(scores_array)
        st.write(f"Shapiro-Wilk statistic: {shapiro_stat:.3f}")
        st.write(f"P-value: {shapiro_p:.3f}")
        st.write(f"*Interpretation:* {'Normal distribution' if shapiro_p > 0.05 else 'Not normal distribution'}")

# Data Export Section
st.markdown("---")
st.header("Data Export & Reports")

# Enhanced data with statistics
enhanced_df = filtered_df.copy()
enhanced_df['Z_Score'] = (enhanced_df['Score'] - avg_score) / std_dev
enhanced_df['Percentile'] = [stats.percentileofscore(scores_array, score) for score in enhanced_df['Score']]
enhanced_df['Performance_Level'] = np.select(
    [enhanced_df['Score'] >= 80, enhanced_df['Score'] >= 60, enhanced_df['Score'] >= 50],
    ['Excellent', 'Good', 'Average'],
    default='Needs Improvement'
)

col13, col14 = st.columns(2)

with col13:
    st.subheader("Enhanced Data Preview")
    st.dataframe(enhanced_df.head(10), use_container_width=True)

with col14:
    st.subheader("Export Options")
    
    # CSV Download
    csv_data = enhanced_df.to_csv(index=False)
    st.download_button(
        label="Download Enhanced CSV",
        data=csv_data,
        file_name="enhanced_tuition_analytics.csv",
        mime="text/csv"
    )
    
    # Excel Download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        enhanced_df.to_excel(writer, sheet_name='Enhanced_Data', index=False)
        summary_stats = pd.DataFrame({
            'Metric': ['Average', 'Median', 'Std Dev', 'Variance', 'Skewness', 'Kurtosis'],
            'Value': [avg_score, median_score, std_dev, variance, skewness, kurtosis]
        })
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    st.download_button(
        label="Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="tuition_analytics_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Footer
st.markdown("---")
st.markdown("### Advanced Tuition Analytics Dashboard")
st.markdown("*Built with:* Streamlit • Pandas • NumPy • Plotly • SciPy • OpenPyXL")
st.success("Dashboard loaded successfully with advanced analytics!")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Info")
st.sidebar.info(f"*Records:* {len(filtered_df)}/{len(df)}  \n*Students:* {filtered_df['Student_ID'].nunique()}  \n*Date Range:* {filtered_df['Test_Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Test_Date'].max().strftime('%Y-%m-%d')}")