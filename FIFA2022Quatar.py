import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import plotly.graph_objects as go
import plotly.figure_factory as ff
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from codecarbon import EmissionsTracker
import random 
import tensorflow as tf

try:
    from tensorflow.python.framework import dtypes
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")

# Load image
image_quatar2022 = Image.open('quatar2022.jpeg')
image_quatar2022_2 = Image.open('2022_FIFA_World_Cup_image_2.jpg')
# Load additional image, audio, and video
image_featured = Image.open('CupImage.jpg')
image_F = Image.open('Image_6.jpg')
image_M = Image.open('Image_7.jpg')
audio_fifa = "k-naan-waving.mp3"
audio_fifa_2 = "shakira-la-la-la.mp3"
audio_fifa_3 = "shakira-waka-waka.mp3"
audio_fifa_4 = "we-are-one-ole-ola.mp3"
audio_fifa_5 = "hayya-hayya-better-together-fifa-world-cup-2022-8d-audio-version-use-headphones-8d-music-song-128-ytshorts.savetube.me.mp3"
audio_1= "sound_effect.mp3"
video_intro = "FIFA_World_Cup_2022_Soundtrack.mp4"
video_concu = "Argentina v France _ FIFA World Cup Qatar 2022.mp4"


st.set_page_config(
    page_title="FIFA World Cup 2022 Data Analysis",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("FIFA World Cup 2022 Data Analysis")
# Initialize session state
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Welcome'

# Styling for sidebar headers
sidebar_header_style = (
    "color: black; background-color: yellow;"
    "text-align: center; border-bottom: 2px solid yellow;"
)

# Sidebar
st.sidebar.markdown("<h2 style='" + sidebar_header_style + "'>Explore FIFA World Cup 2022 Data Analysis</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Navigate through below sections:")

# Page selection buttons
button_labels = ['Welcome 🏠', 'Introduction 📖', 'Visualization 📊', 'Prediction 📈', 'Feature of Importance & Shap 📊', 'MLflow & Deployment 🚀', 'Conclusion 🏁']
selected_button = st.sidebar.radio("Select a page below to explore:", button_labels)

# Set the selected page based on the button clicked
if selected_button == 'Welcome 🏠':
    st.session_state.app_mode = 'Welcome'
elif selected_button == 'Introduction 📖':
    st.session_state.app_mode = 'Introduction'
elif selected_button == 'Visualization 📊':
    st.session_state.app_mode = 'Visualization'
elif selected_button == 'Prediction 📈':
    st.session_state.app_mode = 'Prediction'
elif selected_button == 'Feature of Importance & Shap 📊':
    st.session_state.app_mode = 'Feature of Importance & Shap'
elif selected_button == 'MLflow & Deployment 🚀':
    st.session_state.app_mode = 'MLflow & Deployment'
elif selected_button == 'Conclusion 🏁':
    st.session_state.app_mode = 'Conclusion'

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Add custom font and styling */
        .welcome-text {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 24px;
            color: #fff;
            text-align: center;
            padding: 20px;
            background-color: #17202A; /* Dark blue background */
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        }
        h2 {
            font-size: 36px;
            margin-bottom: 20px;
            color: #FFD700; /* Gold color */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        p {
            font-size: 20px;
            line-height: 1.5;
            color: #fff; /* White color */
            margin-bottom: 15px;
        }
        video {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        }
        /* Style for sidebar */
        .sidebar {
            background-color: #2C3E50; /* Dark sidebar background */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        }
        .sidebar-header {
            color: #FFD700; /* Gold color */
            font-size: 24px;
            margin-bottom: 20px;
        }
        .sidebar-item {
            font-size: 18px;
            color: #fff; /* White color */
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome page content
if st.session_state.app_mode == 'Welcome':
    # Sidebar content for Welcome page

    st.sidebar.markdown("<p style='color: yellow; font-family: Arial, sans-serif;'>Navigate below Welcome sidebar:</p>", unsafe_allow_html=True)

    st.sidebar.markdown("[Welcome](#welcome-section)")

    # Welcome section
    st.markdown(
        """
        <div id="welcome-section" class="welcome-text">
            <h2>Welcome to FIFA World Cup 2022 Data Analysis</h2>
            <p>The FIFA World Cup is the biggest football sports competition where countries from all over the world come together to compete for the most glorious and amazing cup. 🔍 In this app, we're diving into what affects how many goals a team scores in every game during the FIFA World Cup 2022, & Other factors which matters in The Football Match. Why? Well, in football, by scoring more goals often means you're more likely to win the game. Let's explore why that's the case.</p>
            <p style="font-style: italic;">"Football is about scoring goals." - Pep Guardiola</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load the video
    video_path = "Fifa World Cup Opening Shows for Concept K.mp4"
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

    # Disclaimer message (initially hidden)
    if st.sidebar.button("Show Disclaimer"):
        st.sidebar.markdown(
            """
            <div style="font-family: Arial, sans-serif;">
                <p style="font-weight: bold;">⚠️ Disclaimer:</p>
                <p>We're not predicting game winners. Instead, we're analyzing factors that increase goal-scoring likelihood, which also enhances a team's chance of winning.</p>
            </div>
            """,
            unsafe_allow_html=True
    )


# Introduction Page
elif st.session_state.app_mode == 'Introduction':
    st.subheader("Introduction")
    st.sidebar.markdown("<p style='color: yellow; font-family: Arial, sans-serif;'>Navigate below Introduction sidebar:</p>", unsafe_allow_html=True)

    # Welcoming message and image
    st.markdown("<h1 style='text-align: center;'>Habibi, Enjoy FIFA World Cup 2022 Data Analysis App!</h1>", unsafe_allow_html=True)



    st.markdown("<p style='font-family: Arial; font-size: 16px;'>💡 Pro Tip:</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Comic Sans MS; font-size: 20px; color: #FF1493;'>🎵 Enjoy the chosen FIFA World Cup song for you, in the left side bar! 🎉 Feel free to adjust the volume 🔊 or stop the song ⏹️ whenever you want. 🕺💃</p>", unsafe_allow_html=True)

    st.sidebar.subheader("Play FIFA World Cup Song")
    st.sidebar.audio(audio_fifa_3, format='audio/mp3')
    st.video(video_intro, format='video/mp4')

    


    # Set title font, color, and style
    st.markdown(
        """
        <h1 style='text-align: left; color: #1E88E5; font-family: "Arial Black", Gadget, sans-serif;'>
        🎯 Objectives
        </h1>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <p style='text-align: justify; color: #FFFFFF; font-family: "Arial Narrow", sans-serif; font-size: 18px;'>
        <b style='color: #FFFFFF;'>Our mission is to understand which team features affect how many goals teams score in the FIFA World Cup 2022.</b> 
        We're particularly interested in discovering what makes teams score more goals, 
        as well as exploring other factors like possession to learn about team dynamics and strategies.
        </p>
        """,
        unsafe_allow_html=True,
    )



    # Key Variables
    st.markdown("### Key Variables")
    st.markdown(
        "Below are the key variables we emphasize in our analysis, though there are more additional variables considered:",
        unsafe_allow_html=True,
    )
    st.markdown("- <span style='color: #333333;'>Team</span>", unsafe_allow_html=True)
    st.markdown("- <span style='color: #333333;'>Possession</span>", unsafe_allow_html=True)
    st.markdown("- <span style='color: #333333;'>Number of Goals</span>", unsafe_allow_html=True)
    st.markdown("- <span style='color: #333333;'>Corners</span>", unsafe_allow_html=True)
    st.markdown("- <span style='color: #333333;'>On Target Attempts</span>", unsafe_allow_html=True)
    st.markdown("- <span style='color: #333333;'>Defensive Pressures Applied</span>", unsafe_allow_html=True)

    # Description of Data
    st.markdown("### Description of Data")
    
    st.markdown(
        "<p style='color: #333333;'>Let's take a look at some descriptive statistics of the data:</p>",
        unsafe_allow_html=True,
    )



    # Load data
    df = pd.read_csv("FIFAWorldCup2022.csv")

    # Interactive widgets
    st.sidebar.title('Data Exploration Options')

    # Default selection for Team 1
    default_selected_team1 = ['QATAR']

    # Dropdown menu for team selection (Team 1)
    selected_teams_team1 = st.sidebar.multiselect('Select Teams (Team 1)', df['team1'].unique(), default=default_selected_team1)

    # Default selection for Team 2
    default_selected_team2 = ['ECUADOR']

    # Dropdown menu for team selection (Team 2)
    selected_teams_team2 = st.sidebar.multiselect('Select Teams (Team 2)', df['team2'].unique(), default=default_selected_team2)

    # Filter data based on user selections for both teams
    filtered_df_team1 = df[df['team1'].isin(selected_teams_team1)]
    filtered_df_team2 = df[df['team2'].isin(selected_teams_team2)]

    # Combine filtered data for both teams
    filtered_df = pd.concat([filtered_df_team1, filtered_df_team2])

    # Display interactive report
    if st.sidebar.button('Show Report'):
        if filtered_df.empty:
            st.warning("No data available for the selected teams.")
        else:
            # Summary statistics
            st.subheader("Summary Statistics")
            st.write(filtered_df.describe())

            # Bar chart for number of goals
            st.subheader("Number of Goals Comparison")
            fig_goals = px.bar(filtered_df, x='team1', y='number of goals team1', color='team1', title='Number of Goals Comparison')
            st.plotly_chart(fig_goals)

            # Histogram for possession
            st.subheader("Distribution of Possession")
            fig_possession = px.histogram(filtered_df, x='possession team1', color='team1', nbins=20, title='Possession Distribution')
            st.plotly_chart(fig_possession)


            # Line plot for trends over time (assuming 'date' column represents time)
            if 'date' in filtered_df.columns:
                st.subheader("Trends Over Time")
                fig_trends = px.line(filtered_df, x='date', y='possession team1', color='team1', title='Possession Over Time')
                st.plotly_chart(fig_trends)

            # Additional statistics and insights
            st.subheader("Additional Statistics and Insights")
            # Remove percentage symbols and convert to numeric
            filtered_df['possession team1'] = filtered_df['possession team1'].str.replace('%', '').astype(float)

            # Create a bar plot for total goals scored, average possession, and average number of goals per game
            fig, ax = plt.subplots(figsize=(10, 6))

            # Total goals scored
            total_goals = filtered_df['number of goals team1'].sum()
            ax.bar("Total Goals Scored", total_goals, color='blue')
            ax.text("Total Goals Scored", total_goals, f'{total_goals}', ha='center', va='bottom')

            # Average possession
            avg_possession = filtered_df['possession team1'].mean()
            ax.bar("Average Possession", avg_possession, color='green')
            ax.text("Average Possession", avg_possession, f'{avg_possession:.2f}%', ha='center', va='bottom')

            # Average number of goals per game
            avg_goals_per_game = filtered_df['number of goals team1'].mean()
            ax.bar("Average Goals Per Game", avg_goals_per_game, color='orange')
            ax.text("Average Goals Per Game", avg_goals_per_game, f'{avg_goals_per_game:.2f}', ha='center', va='bottom')

            # Set labels and title
            ax.set_ylabel('Value')
            ax.set_title('Comparison of Statistics')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Display the plot
            st.pyplot(fig)
            # Dynamic data exploration for team 1
            st.subheader("Dynamic Data Exploration (Team 1)")
            st.write(filtered_df_team1)

            # Dynamic data exploration for team 2
            st.subheader("Dynamic Data Exploration (Team 2)")
            st.write(filtered_df_team2)

    # User Feedback Integration
    st.sidebar.title('User Feedback')
    user_email = st.sidebar.text_input("Enter your email address:")
    feedback = st.sidebar.text_area("Please provide your feedback here:")
    submit_button = st.sidebar.button("Submit Feedback")
    if submit_button:
        # Store feedback in a file or database
        with open("feedback.txt", "a") as f:
            f.write("Email: {}\nFeedback: {}\n".format(user_email, feedback))
        st.sidebar.success("Thank you for your feedback!")

        # Send feedback to email
        sender_email = user_email  # Use user's email as sender
        receiver_emails = ["jackson.mukeshimana@nyu.edu", "mukesjackson02@gmail.com"]  # Update with receiver emails

        # Compose email
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(receiver_emails)
        message["Subject"] = "User Feedback"

        # Add message body
        message.attach(MIMEText("User Email: {}\n\nFeedback: {}".format(user_email, feedback), "plain"))

        # Connect to SMTP server and send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.sendmail(sender_email, receiver_emails, message.as_string())

        st.sidebar.success("Your feedback has been submitted and sent to the admins.")
    # Convert categorical columns to numeric codes
    df['team1'] = df['team1'].astype('category').cat.codes
    df['team2'] = df['team2'].astype('category').cat.codes

    # Remove percentage signs and convert to numeric
    columns_to_convert = ['possession team1', 'possession team2', 'possession in contest']
    for column in columns_to_convert:
        df[column] = df[column].astype(str).str.rstrip('%').astype(float)

    # Convert converted columns to categorical codes
    for column in columns_to_convert:
        df[column] = df[column].astype('category').cat.codes

    # Convert other categorical columns to numeric codes
    columns_to_convert_to_codes = ['date', 'hour', 'category']
    for column in columns_to_convert_to_codes:
        df[column] = df[column].astype('category').cat.codes
    # Display summary statistics
    st.dataframe(df.describe())

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Missing Values
    st.markdown("### Missing Values")
    st.markdown("Let's examine the presence of missing values in our dataset:")

    # Calculate percentage of missing values for each column
    missing_values = df.isnull().sum() / len(df) * 100

    # Display missing value percentages
    st.write("Percentage of missing values for each column:")
    st.write(missing_values)

    # Assess overall completeness of the dataset
    completeness_ratio = df.notnull().sum().sum() / (len(df) * len(df.columns))
    
    st.write(f"Overall completeness ratio: {completeness_ratio:.2f}")
    
    if completeness_ratio >= 0.85:
        st.success("The dataset has a high level of completeness, providing us with reliable data for analysis.")
    else:
        st.warning("The dataset has a low level of completeness, which may affect the reliability of our analysis.")

    
    
    # Conclusion
    st.markdown("### Recap")
    
    # Describe conclusion with concise and clear language
    st.markdown(
        """
        <p style='color: #FFFFFF; font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;'>
        In this dashboard, we explored the FIFA World Cup 2022 dataset, focusing on key variables like possession, number of goals, corners, and defensive pressures. 
        We also assessed the cleanliness of our dataset for reliability and usability.
        </p>
        """,
        unsafe_allow_html=True,
    )



elif st.session_state.app_mode == 'Visualization':
    # Play FIFA World Cup song
    st.sidebar.subheader("Play FIFA World Cup Song")
    st.sidebar.markdown("<p style='font-family: Impact; font-size: 16px; color: #007ACC;'>🎵 Enjoy the below chosen FIFA World Cup song for you! 🎶 Feel free to adjust the volume or stop the song whenever you want. 🎧</p>", unsafe_allow_html=True)

    st.sidebar.audio(audio_fifa_2, format='audio/mp3')



    # Left sidebar for text
    st.subheader("Explore visualizations of the FIFA World Cup 2022 data.")
    st.image(image_quatar2022, width=800)

    # Load the FIFA World Cup 2022 dataset
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Convert categorical columns to numeric codes
    df['team1'] = df['team1'].astype('category').cat.codes
    df['team2'] = df['team2'].astype('category').cat.codes

    # Remove percentage signs and convert to numeric
    columns_to_convert = ['possession team1', 'possession team2', 'defensive pressures applied team1', 'passes team2', 'passes completed team2', 'on target attempts team2', 'inbehind offers to receive team2', 'attempted defensive line breaks team2']
    for column in columns_to_convert:
        df[column] = df[column].astype(str).str.rstrip('%').astype(float)

    # Convert converted columns to categorical codes
    for column in columns_to_convert:
        df[column] = df[column].astype('category').cat.codes

    # Convert other categorical columns to numeric codes
    columns_to_convert_to_codes = ['date', 'hour', 'category']
    for column in columns_to_convert_to_codes:
        df[column] = df[column].astype('category').cat.codes

    # Select independent variables and target variable
    selected_features = ["assists team2", "attempted defensive line breaks team2", "on target attempts team2", "inbehind offers to receive team2", "possession team2", "passes completed team2", "number of goals team2"]

    # Extract selected columns from the dataset
    df_selected = df[selected_features]

    # Calculate the correlation matrix
    corr_matrix = df_selected.corr()

    # Plot the heatmap using Streamlit
    st.write("## Correlation Heatmap")

    st.markdown(
        "<h3 style='text-align: center; color: #1E88E5;'>This heatmap illustrates the correlation between selected variables and the number of goals scored by Team 2.</h3>",
        unsafe_allow_html=True,
    )


    # Add color palette customization
    color_palette = st.selectbox("Select color palette:", ["coolwarm", "viridis", "magma", "inferno", "plasma"], index=0)
    cmap = sns.color_palette(color_palette, as_cmap=True)

    # Plot the heatmap
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", ax=ax, square=True, linewidths=0.5, linecolor='black')

    # Set labels and title
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    ax.set_title("Correlation Heatmap")

    # Show plot
    st.pyplot(fig)



    # Define default values for independent and dependent variables
    scatter_independent_default = 'corners team1'
    scatter_dependent_default = 'number of goals team1'

    # Interactive Selection: Allow users to select specific variables
    independent_variable_scatter = st.selectbox("Select Independent Variable", df.columns[:-1], index=df.columns.get_loc(scatter_independent_default) if scatter_independent_default in df.columns else 0, key='scatter_independent')
    dependent_variable_scatter = st.selectbox("Select Dependent Variable", df.columns[:-1], index=df.columns.get_loc(scatter_dependent_default) if scatter_dependent_default in df.columns else 0, key='scatter_dependent')

    # Check if the selected columns exist and are numeric
    if independent_variable_scatter in df.columns and dependent_variable_scatter in df.columns:
        # Convert selected columns to numeric types, ignoring non-numeric values
        df[independent_variable_scatter] = pd.to_numeric(df[independent_variable_scatter], errors='coerce')
        df[dependent_variable_scatter] = pd.to_numeric(df[dependent_variable_scatter], errors='coerce')

        # Drop rows with NaN values in the selected columns after conversion
        df.dropna(subset=[independent_variable_scatter, dependent_variable_scatter], inplace=True)

        # Color Palette Customization: Allow users to choose different color palettes
        palette_scatter = st.radio("Select Color Palette", ["viridis", "magma", "plasma", "inferno", "coolwarm"], key='color_palette')

        # Create scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=independent_variable_scatter, y=dependent_variable_scatter, ax=ax, palette=palette_scatter)
        ax.set_xlabel(independent_variable_scatter)
        ax.set_ylabel(dependent_variable_scatter)
        ax.set_title(f'{dependent_variable_scatter} vs {independent_variable_scatter}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show interactive sorting option
        if st.checkbox('Sort Variables'):
            sort_order = st.radio("Select sort order", ["ascending", "descending"], index=1)
            if sort_order == "ascending":
                df_sorted = df.sort_values(by=independent_variable_scatter)
            else:
                df_sorted = df.sort_values(by=independent_variable_scatter, ascending=False)
            st.dataframe(df_sorted)

        # Show the scatter plot with a descriptive title
        st.write(f"## Scatter Plot: {dependent_variable_scatter} vs {independent_variable_scatter}")
        st.pyplot(fig)

    else:
        st.write("Selected columns not found in DataFrame or are not numeric.")


    # Histogram
    st.subheader('Histogram')

    # Interactive Selection: Allow users to select specific variables
    hist_default = 'passes completed team1'
    independent_variable_hist = st.selectbox("Select Variable", df.columns[:-1], index=df.columns.get_loc(hist_default) if hist_default in df.columns else 0, key='hist_independent')

    # Create histogram
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(data=df, x=independent_variable_hist, ax=ax_hist)
    ax_hist.set_xlabel(independent_variable_hist)
    ax_hist.set_title(f'Histogram of {independent_variable_hist}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show interactive sorting option
    if st.checkbox('Sort Values'):
        sort_order_hist = st.radio("Select sort order", ["ascending", "descending"], index=1, key='hist_sort_order')
        if sort_order_hist == "ascending":
            df_sorted_hist = df.sort_values(by=independent_variable_hist)
        else:
            df_sorted_hist = df.sort_values(by=independent_variable_hist, ascending=False)
        st.dataframe(df_sorted_hist)

    # Color Palette Customization: Allow users to choose different color palettes
    palette = st.radio("Select Color Palette", ["viridis", "magma", "plasma", "inferno", "coolwarm"], key='hist_color_palette')
    sns.set_palette(palette)
    st.pyplot(fig_hist)

    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Example DataFrame
    # You need to replace this with your actual DataFrame
    data = {
        'free kicks team2': [1, 2, 3, 4, 5],
        'number of goals team2': [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)

    # Define the play_sound function
    def play_sound():
        st.audio("sound_effect.mp3", format="audio/mp3")

    # Define the function to generate the box plot
    def generate_box_plot(selected_variable):
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df, x=selected_variable, ax=ax_box)
        ax_box.set_xlabel(selected_variable)
        ax_box.set_title(f'Box Plot of {selected_variable}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_box)

    # Example variable definitions
    index_box = 0  # Example index, replace with your logic
    independent_variable_box = 'free kicks team2'  # Example variable, replace with your logic

    # Check if the index is within the range of options
    if index_box < len(df.columns[:-1]):
        # Call the function to generate the box plot
        generate_box_plot(independent_variable_box)

        # Color Palette Customization: Allow users to choose different color palettes
        color_palette_key = 'color_palette_box'
        color_palette = st.selectbox("Select color palette:", ["coolwarm", "viridis", "magma", "inferno", "plasma"], index=0, key=color_palette_key)
        cmap = sns.color_palette(color_palette, as_cmap=True)

        # Show interactive sorting option
        sort_values_checkbox_key = 'sort_values_checkbox_box'
        if st.checkbox('Sort Values', key=sort_values_checkbox_key):
            sort_order_box = st.radio("Select sort order", ["ascending", "descending"], index=1, key='sort_order_box')
            if sort_order_box == "ascending":
                df_sorted_box = df.sort_values(by=independent_variable_box)
            else:
                df_sorted_box = df.sort_values(by=independent_variable_box, ascending=False)
            st.dataframe(df_sorted_box)

        # Dynamic Thresholding: Allow users to adjust threshold for displaying box plot
        min_val = float(df[independent_variable_box].min())  # Cast min value to float
        max_val = float(df[independent_variable_box].max())  # Cast max value to float
        mean_val = df[independent_variable_box].mean()  # No need to cast mean value, it's already float
        threshold_box = st.slider('Threshold for Box Plot', min_value=min_val, max_value=max_val, value=mean_val)

        # Play sound effect when plot is generated
        if st.button("Generate Box Plot"):
            play_sound()

    else:
        st.warning("No valid variable selected for the box plot.")

   # Bar plot
    st.subheader('Bar Plot')
    
    # Interactive Selection: Allow users to select specific variables
    bar_independent_default = 'free kicks team2'
    bar_dependent_default = 'number of goals team2'
    
    # Safe selection of default indices
    def get_safe_index(columns, default_value):
        try:
            return list(columns).index(default_value) if default_value in columns else 0
        except (ValueError, TypeError):
            return 0
    
    # Create selectboxes with safe index selection
    independent_variable_bar = st.selectbox(
        "Select Independent Variable",
        options=df.columns[:-1],
        index=get_safe_index(df.columns[:-1], bar_independent_default),
        key='bar_independent'
    )
    
    dependent_variable_bar = st.selectbox(
        "Select Dependent Variable",
        options=df.columns[:-1],
        index=get_safe_index(df.columns[:-1], bar_dependent_default),
        key='bar_dependent'
    )
    
    # Color Palette Customization: Allow users to choose different color palettes
    palette_bar = st.radio("Select Color Palette", ["viridis", "magma", "plasma", "inferno", "coolwarm"], key='bar_color_palette')
    
    # Create bar plot function
    def create_bar_plot(data, x, y, palette):
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(data=data, x=x, y=y, ax=ax_bar, palette=palette)
        ax_bar.set_xlabel(x)
        ax_bar.set_ylabel(y)
        ax_bar.set_title(f'Bar Plot of {y} vs {x}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_bar)
    
    # Display initial bar plot
    create_bar_plot(df, independent_variable_bar, dependent_variable_bar, palette_bar)
    
    # Show interactive sorting option
    if st.checkbox('Sort Values', key='sort_checkbox'):
        sort_order_bar = st.radio("Select sort order", ["ascending", "descending"], index=1, key='sort_order_radio')
        if sort_order_bar == "ascending":
            df_sorted_bar = df.sort_values(by=dependent_variable_bar)
        else:
            df_sorted_bar = df.sort_values(by=dependent_variable_bar, ascending=False)
        create_bar_plot(df_sorted_bar, independent_variable_bar, dependent_variable_bar, palette_bar)
    
    # Dynamic Thresholding: Allow users to adjust threshold for displaying bar plot
    min_value_bar = float(df[dependent_variable_bar].min())  # Convert min_value to float
    max_value_bar = float(df[dependent_variable_bar].max())  # Convert max_value to float
    value_bar = float(df[dependent_variable_bar].mean())     # Convert value to float
    step_bar = 0.1  # Set step as a float
    threshold_bar = st.slider('Threshold for Bar Plot', min_value=min_value_bar, max_value=max_value_bar, value=value_bar, step=step_bar, key='threshold_slider')
    
    # Play sound effect when plot is generated
    if st.button("Generate Bar Plot"):
        play_sound()
    import random  # Import the random module

    # Additional graphs
    st.subheader('Additional Graphs')
    st.markdown("Extra graphs based on picked variables from the dataset (Pick Yours to Explore as well!):")
    
    # Define default values
    additional_independent_default_1 = 'on target attempts team1'
    additional_dependent_default_1 = 'number of goals team1'
    additional_independent_default_2 = 'assists team2'
    additional_dependent_default_2 = 'number of goals team2'
    
    # Safe selection of default indices function
    def get_safe_index(columns, default_value):
        try:
            return list(columns).index(default_value) if default_value in columns else 0
        except (ValueError, TypeError):
            return 0
    
    # Plot 1: First set of selectboxes with safe index selection
    additional_independent_variable_1 = st.selectbox(
        "Select Independent Variable (Plot 1)", 
        options=df.columns[:-1],
        index=get_safe_index(df.columns[:-1], additional_independent_default_1),
        key='additional_independent_1'
    )
    
    additional_dependent_variable_1 = st.selectbox(
        "Select Dependent Variable (Plot 1)", 
        options=df.columns[:-1],
        index=get_safe_index(df.columns[:-1], additional_dependent_default_1),
        key='additional_dependent_1'
    )
    
    # Create scatter plot
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x=additional_independent_variable_1, y=additional_dependent_variable_1, ax=ax1)
    ax1.set_xlabel(additional_independent_variable_1)
    ax1.set_ylabel(additional_dependent_variable_1)
    ax1.set_title(f'{additional_dependent_variable_1} vs {additional_independent_variable_1}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Dynamic Thresholding for scatter plot
    min_val_1 = float(df[additional_dependent_variable_1].min())
    max_val_1 = float(df[additional_dependent_variable_1].max())
    mean_val_1 = float(df[additional_dependent_variable_1].mean())
    step_val_1 = (max_val_1 - min_val_1) / 100  # Adjust step value
    threshold_scatter_1 = st.slider('Threshold for Scatter Plot', min_value=min_val_1, max_value=max_val_1, value=mean_val_1, step=step_val_1, format="%.2f")
    
    st.pyplot(fig1)
    
    # Plot 2: Second set of selectboxes with safe index selection
    additional_independent_variable_2 = st.selectbox(
        "Select Independent Variable (Plot 2)", 
        options=df.columns[:-1],
        index=get_safe_index(df.columns[:-1], additional_independent_default_2),
        key='additional_independent_2'
    )
    
    additional_dependent_variable_2 = st.selectbox(
        "Select Dependent Variable (Plot 2)", 
        options=df.columns[:-1],
        index=get_safe_index(df.columns[:-1], additional_dependent_default_2),
        key='additional_dependent_2'
    )
    
    # Check if data for the selected variables is available and create line plot
    if additional_independent_variable_2 in df.columns and additional_dependent_variable_2 in df.columns:
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=df, x=additional_independent_variable_2, y=additional_dependent_variable_2, ax=ax2)
        ax2.set_xlabel(additional_independent_variable_2)
        ax2.set_ylabel(additional_dependent_variable_2)
        ax2.set_title(f'{additional_dependent_variable_2} vs {additional_independent_variable_2}')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
        # Display the plot
        st.pyplot(fig2)
    else:
        st.warning("Selected variables not found in the dataset. Please make sure to select valid variables.")
    
    # Interactive Sorting
    if st.checkbox('Interactive Sorting'):
        sort_order = st.radio("Select sort order", ["ascending", "descending"], index=1)
        if sort_order == "ascending":
            df_sorted = df.sort_values(by=additional_independent_variable_2)
        else:
            df_sorted = df.sort_values(by=additional_independent_variable_2, ascending=False)
        st.dataframe(df_sorted)
    
        # Additional Fun Facts
        show_fun_facts = st.button("Show Fun Facts")
        if show_fun_facts:
            st.session_state.show_fun_facts = True
    
        if "show_fun_facts" not in st.session_state:
            st.session_state.show_fun_facts = False
    
        if st.session_state.show_fun_facts:
            with st.expander("", expanded=True):
                expander_title = "<h2 style='font-family: Arial; font-size: 20px;'>Additional Fun Facts</h2>"
                st.markdown(expander_title, unsafe_allow_html=True)
                fun_facts = [
                    "The fastest goal in FIFA World Cup history was scored by Hakan Şükür of Turkey in 2002, just 11 seconds into the match!",
                    "Brazil holds the record for the most FIFA World Cup titles, with a total of 5 wins.",
                    "The 2022 FIFA World Cup final was held at the Lusail Iconic Stadium in Qatar, which has a seating capacity of 80,000 people."
                ]
    
                # Display additional fun facts with numbers and bold formatting
                for i, fact in enumerate(fun_facts, 1):
                    st.markdown(f"<p style='font-family: Arial; font-size: 16px; color: #6495ED;'><strong>{i}. </strong>{fact}</p>", unsafe_allow_html=True)

    # Conclusion and Surprise Element
    show_conclusion = st.button("Show Conclusion and Surprise Element")
    if show_conclusion:
        st.subheader("Conclusion")
        st.write("Congratulations! You've explored a variety of visualizations and interactive features to gain insights from the FIFA World Cup 2022 dataset. But wait, there's more!")

        # Surprise Element: Random Fun Fact
        random_fact = "Did you know that the FIFA World Cup trophy weighs about 6.175 kilograms (13.61 pounds)?"
        st.markdown(f"<p style='font-family: Georgia; color: #FF0000;'>Here's a random fun fact: {random_fact}</p>", unsafe_allow_html=True)





elif st.session_state.app_mode == 'Prediction':
    st.subheader("Prediction")
    st.sidebar.subheader("Play FIFA World Cup Song")


    st.sidebar.markdown("<p style='font-family: Impact; font-size: 16px; color: #007ACC;'>🎵 Enjoy the below chosen FIFA World Cup song for you! 🎶 Feel free to adjust the volume or stop the song whenever you want. 🎧</p>", unsafe_allow_html=True)

    st.sidebar.audio(audio_fifa_4, format='audio/mp3')
    st.image(image_featured, use_column_width=True)
    st.title("FIFA World Cup 2022 Data Analysis - Prediction")
    st.markdown("Select a machine learning model and variables to predict outcomes.")

    # Load the dataset
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Convert categorical columns to numeric codes
    df['team1'] = df['team1'].astype('category').cat.codes
    df['team2'] = df['team2'].astype('category').cat.codes

    # Remove percentage signs and convert to numeric
    columns_to_convert = ['possession team1', 'possession team2', 'possession in contest']
    for column in columns_to_convert:
        df[column] = df[column].astype(str).str.rstrip('%').astype(float)

    # Convert converted columns to categorical codes
    for column in columns_to_convert:
        df[column] = df[column].astype('category').cat.codes

    # Convert other categorical columns to numeric codes
    columns_to_convert_to_codes = ['date', 'hour', 'category']
    for column in columns_to_convert_to_codes:
        df[column] = df[column].astype('category').cat.codes

    # Set dependent variable
    selected_target = 'number of goals team2'

    # Set default independent variables
    default_independent_variables = ["assists team2", "attempts inside the penalty area team2", "offsides team2"]

    # Calculate correlation with the dependent variable
    corr_with_target = df.corr()[selected_target].abs()

    # Filter out independent variables with correlation > 0.1
    filtered_features = corr_with_target[corr_with_target > 0.1].index.tolist()

    # Ensure default values exist in available options
    default_features = [feat for feat in default_independent_variables if feat in filtered_features]

    # Features and target variable selection
    selected_features = st.multiselect("Select Independent Variables", filtered_features, default=default_features)

    # Machine learning model selection
    selected_models = st.multiselect("Select Model(s)", ['Linear Regression', 'Random Forest', 'Gradient Boosting'], default=['Linear Regression'])

    # Custom hyperparameters for selected models
    custom_hyperparameters = {}
    for model in selected_models:
        if model == 'Random Forest':
            custom_hyperparameters['Random Forest'] = {
                'n_estimators': st.number_input("Number of Estimators (Random Forest)", min_value=10, max_value=1000, value=100, step=10)
            }
        elif model == 'Gradient Boosting':
            custom_hyperparameters['Gradient Boosting'] = {
                'n_estimators': st.number_input("Number of Estimators (Gradient Boosting)", min_value=10, max_value=1000, value=100, step=10),
                'learning_rate': st.number_input("Learning Rate (Gradient Boosting)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            }

    if not selected_features:
        st.warning("Please select at least one independent variable.")
    else:
        # Extract selected columns from the dataset
        df_selected = df[selected_features + [selected_target]]

        # Remove rows with missing values
        df_selected = df_selected.dropna()

        if df_selected.empty:
            st.warning("No data available after removing rows with missing values. Please choose different variables.")
        else:
            # Check if selected variables have numeric data
            numeric_columns = df_selected.select_dtypes(include=['float', 'int']).columns

            if len(numeric_columns) != len(selected_features) + 1:  # Check if all selected variables are numeric
                non_numeric_variables = [var for var in selected_features + [selected_target] if var not in numeric_columns]
                st.error(f"The following selected variables contain non-numeric values: {', '.join(non_numeric_variables)}")
            else:
                X = df_selected[selected_features]
                y = df_selected[selected_target]

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Guiding message
                st.info("Select a machine learning model.")

                # Display dependent variable with enhanced style
                st.markdown(f"<p style='font-size: 18px; color: #3366ff; font-weight: bold;'>Dependent Variable to Predict: {selected_target}</p>", unsafe_allow_html=True)

                for model in selected_models:
                    st.subheader(f"{model} Model")

                    if model == 'Linear Regression':
                        # Linear Regression model implementation
                        try:
                            # Train the model
                            model = LinearRegression()
                            model.fit(X_train, y_train)

                            # Make predictions
                            y_pred = model.predict(X_test)

                            # Evaluate the model
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            # Model Performance Visualization
                            st.subheader("Model Performance Visualization")

                            # Create histogram data
                            hist_data = [y_test, y_pred]
                            group_labels = ['Actual', 'Predicted']

                            # Create the histogram using Plotly
                            fig_pred_actual_hist = ff.create_distplot(hist_data, group_labels, bin_size=0.5, colors=['blue', 'orange'])

                            # Update layout with enhanced features
                            fig_pred_actual_hist.update_layout(
                                title='Predicted vs Actual Histogram',
                                xaxis_title='Values',
                                yaxis_title='Frequency',
                                showlegend=True,
                                plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Set background color
                                template='plotly_white',  # Use white template for better contrast
                                width=800,  # Increase default width of the chart
                                height=600,  # Increase default height of the chart
                            )

                            # Add buttons for interactivity
                            fig_pred_actual_hist.update_layout(
                                updatemenus=[
                                    {
                                        'buttons': [
                                            {
                                                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                                                'label': 'Play',
                                                'method': 'animate'
                                            },
                                            {
                                                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                                                'label': 'Pause',
                                                'method': 'animate'
                                            }
                                        ],
                                        'direction': 'left',
                                        'pad': {'r': 10, 't': 10},
                                        'showactive': False,
                                        'type': 'buttons',
                                        'x': 0.05,  # Adjust position of the buttons
                                        'xanchor': 'right',
                                        'y': 1.1,  # Adjust position of the buttons
                                        'yanchor': 'top'
                                    },
                                    {
                                        'buttons': [
                                            {'args': [None, {'xaxis': {'type': 'linear'}, 'yaxis': {'type': 'linear'}}], 'label': 'Reset Zoom', 'method': 'relayout'}
                                        ],
                                        'direction': 'down',
                                        'showactive': False,
                                        'type': 'buttons',
                                        'x': 0.05,  # Adjust position of the buttons
                                        'xanchor': 'right',
                                        'y': 1.05,  # Adjust position of the buttons
                                        'yanchor': 'top'
                                    }
                                ]
                            )

                            # Display the histogram
                            st.plotly_chart(fig_pred_actual_hist)



                            # Display model performance metrics
                            st.subheader("Model Performance Metrics")
                            st.write(f"{model} Model Performance:")
                            st.write(f"R-squared: {r2:.2f}")
                            st.write(f"Mean Squared Error: {mse:.2f}")

                            st.write("Interpretation:")
                            if r2 >= 0.7:
                                st.info(f"R-squared of {r2:.2f} shows that the model explains a large proportion of the variance in the dependent variable, indicating a strong relationship between the selected features and the number of goals of the team.")
                            elif r2 >= 0.5:
                                st.warning(f"R-squared of {r2:.2f} shows that the model explains a moderate proportion of the variance in the dependent variable, suggesting a moderate relationship between the selected features and the number of goals of the team.")
                            else:
                                st.error(f"R-squared of {r2:.2f} shows that the model does not explain much of the variance in the dependent variable, indicating a weak relationship between the selected features and the number of goals of the team.")

                            # Check if R-squared is less than zero
                            if r2 < 0:
                                st.error("R-squared is less than zero. There may be an issue with the chosen variable in the dataset. Please consider removing this variable.")

                        except ValueError as e:
                            st.error(f"Error: {e}. Please ensure all selected variables are numeric.")

                    elif model == 'Random Forest':
                        # Random Forest model implementation with custom hyperparameters
                        try:
                            # Train the model with custom hyperparameters
                            n_estimators = custom_hyperparameters['Random Forest']['n_estimators']
                            model = RandomForestRegressor(n_estimators=n_estimators)
                            model.fit(X_train, y_train)

                            # Make predictions
                            y_pred = model.predict(X_test)

                            # Evaluate the model
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            # Create scatter plot data
                            scatter_data = go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual', marker=dict(color='orange'))

                            # Create perfect prediction line data
                            perfect_line = go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction', line=dict(color='blue', dash='dash'))

                            # Create the figure
                            fig_rf = go.Figure(data=[scatter_data, perfect_line])

                            # Update layout with enhanced features
                            fig_rf.update_layout(
                                title='Random Forest: Predicted vs Actual',
                                xaxis_title='Actual',
                                yaxis_title='Predicted',
                                showlegend=True,
                                plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Set background color
                                xaxis=dict(showgrid=True, gridcolor='lightgray'),  # Show gridlines on x-axis
                                yaxis=dict(showgrid=True, gridcolor='lightgray'),  # Show gridlines on y-axis
                                hovermode='closest',  # Set hover mode to show closest data point
                                template='plotly_white',  # Use white template for better contrast
                                width=900,  # Increase default width of the chart
                                height=700,  # Increase default height of the chart
                            )

                            # Add buttons for interactivity
                            fig_rf.update_layout(
                                updatemenus=[
                                    {
                                        'buttons': [
                                            {'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'},
                                            {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}
                                        ],
                                        'direction': 'left',
                                        'pad': {'r': 10, 't': 10},
                                        'showactive': False,
                                        'type': 'buttons',
                                        'x': 0.05,  # Adjust position of the buttons
                                        'xanchor': 'right',
                                        'y': 1.1,  # Adjust position of the buttons
                                        'yanchor': 'top'
                                    },
                                    {
                                        'buttons': [
                                            {'args': [None, {'xaxis': {'type': 'linear'}, 'yaxis': {'type': 'linear'}}], 'label': 'Reset Zoom', 'method': 'relayout'}
                                        ],
                                        'direction': 'down',
                                        'showactive': False,
                                        'type': 'buttons',
                                        'x': 0.05,  # Adjust position of the buttons
                                        'xanchor': 'right',
                                        'y': 1.05,  # Adjust position of the buttons
                                        'yanchor': 'top'
                                    }
                                ]
                            )

                            # Display the scatter plot
                            st.plotly_chart(fig_rf)


                            # Display model performance metrics
                            st.subheader("Model Performance Metrics")
                            st.write(f"{model} Model Performance:")
                            st.write(f"R-squared: {r2:.2f}")
                            st.write(f"Mean Squared Error: {mse:.2f}")

                            st.write("Interpretation:")
                            if r2 >= 0.7:
                                st.info(f"R-squared of {r2:.2f} shows that the model explains a large proportion of the variance in the dependent variable, indicating a strong relationship between the selected features and the number of goals of the team.")
                            elif r2 >= 0.5:
                                st.warning(f"R-squared of {r2:.2f} shows that the model explains a moderate proportion of the variance in the dependent variable, suggesting a moderate relationship between the selected features and the number of goals of the team.")
                            else:
                                st.error(f"R-squared of {r2:.2f} shows that the model does not explain much of the variance in the dependent variable, indicating a weak relationship between the selected features and the number of goals of the team.")

                            # Check if R-squared is less than zero
                            if r2 < 0:
                                st.error("R-squared is less than zero. There may be an issue with the chosen variable in the dataset. Please consider removing this variable.")

                        except ValueError as e:
                            st.error(f"Error: {e}. Please ensure all selected variables are numeric.")

                    elif model == 'Gradient Boosting':
                        # Gradient Boosting model implementation with custom hyperparameters
                        try:
                            # Train the model with custom hyperparameters
                            n_estimators = custom_hyperparameters['Gradient Boosting']['n_estimators']
                            learning_rate = custom_hyperparameters['Gradient Boosting']['learning_rate']
                            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
                            model.fit(X_train, y_train)

                            # Make predictions
                            y_pred = model.predict(X_test)

                            # Evaluate the model
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)




                            # Create a 3D scatter plot using Plotly
                            fig = go.Figure(data=[go.Scatter3d(
                                x=y_test,
                                y=y_pred,
                                z=X_test['assists team2'],  # Use a feature as the third dimension for added insight
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color=X_test['assists team2'],  # Color points by a feature
                                    colorscale='Viridis',  # Choose a color scale
                                    opacity=0.8
                                )
                            )])

                            # Update layout for better presentation
                            fig.update_layout(
                                scene=dict(
                                    xaxis_title='Actual Goals of Team',
                                    yaxis_title='Predicted Goals of Team',
                                    zaxis_title='Assists of Team',  # Update axis titles
                                    bgcolor='rgba(139, 69, 19, 0.8)',  # Set brown background color of the 3D scene
                                ),
                                title=dict(text='Gradient Boosting: Actual vs Predicted Goals of Teams', y=0.95),  # Adjust title position
                                paper_bgcolor='rgba(139, 69, 19, 0.8)',  # Set brown background color of the plot area
                                plot_bgcolor='rgba(139, 69, 19, 0.8)',  # Set brown background color of the plot
                                width=900,  # Increase default width of the chart
                                height=700,  # Increase default height of the chart
                                margin=dict(l=0, r=0, b=50, t=50),  # Adjust margins for better layout
                            )

                            # Add buttons for interactivity
                            fig.update_layout(
                                updatemenus=[
                                    {
                                        'buttons': [
                                            {'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'},
                                            {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}
                                        ],
                                        'direction': 'left',
                                        'pad': {'r': 10, 't': 10},
                                        'showactive': False,
                                        'type': 'buttons',
                                        'x': 0.05,  # Adjust position of the buttons
                                        'xanchor': 'right',
                                        'y': 0.9,  # Adjust position of the buttons
                                        'yanchor': 'top'
                                    },
                                    {
                                        'buttons': [
                                            {'args': [None, {'xaxis': {'type': 'linear'}, 'yaxis': {'type': 'linear'}}], 'label': 'Reset Zoom', 'method': 'relayout'}
                                        ],
                                        'direction': 'down',
                                        'showactive': False,
                                        'type': 'buttons',
                                        'x': 0.05,  # Adjust position of the buttons
                                        'xanchor': 'right',
                                        'y': 0.85,  # Adjust position of the buttons
                                        'yanchor': 'top'
                                    }
                                ]
                            )

                            # Display the plot
                            st.plotly_chart(fig)



                            # Display model performance metrics
                            st.subheader("Model Performance Metrics")
                            st.write(f"{model} Model Performance:")
                            st.write(f"R-squared: {r2:.2f}")
                            st.write(f"Mean Squared Error: {mse:.2f}")

                            st.write("Interpretation:")
                            if r2 >= 0.7:
                                st.info(f"R-squared of {r2:.2f} shows that the model explains a large proportion of the variance in the dependent variable, indicating a strong relationship between the selected features and the number of goals of the team.")
                            elif r2 >= 0.5:
                                st.warning(f"R-squared of {r2:.2f} shows that the model explains a moderate proportion of the variance in the dependent variable, suggesting a moderate relationship between the selected features and the number of goals of the team.")
                            else:
                                st.error(f"R-squared of {r2:.2f} shows that the model does not explain much of the variance in the dependent variable, indicating a weak relationship between the selected features and the number of goals of the team.")

                            # Check if R-squared is less than zero
                            if r2 < 0:
                                st.error("R-squared is less than zero. There may be an issue with the chosen variable in the dataset. Please consider removing this variable.")

                        except ValueError as e:
                            st.error(f"Error: {e}. Please ensure all selected variables are numeric.")


elif st.session_state.app_mode == 'Feature of Importance & Shap':
    st.subheader("Features of Importance & Shap")
    st.sidebar.subheader("Play FIFA World Cup Song")
    st.sidebar.markdown("<p style='font-family: Impact; font-size: 16px; color: #007ACC;'>🎵 Enjoy the below chosen FIFA World Cup song for you! 🎶 Feel free to adjust the volume or stop the song whenever you want. 🎧</p>", unsafe_allow_html=True)

    st.sidebar.audio(audio_fifa_5, format='audio/mp3')

    st.image(image_F, width=800)

    st.title("Feature of Importance & Shap")
    df = pd.read_csv('FIFAWorldCup2022.csv')

    # Preprocess the data
    df['team1'] = df['team1'].astype('category').cat.codes
    df['team2'] = df['team2'].astype('category').cat.codes
    columns_to_convert = ['possession team1', 'possession team2', 'possession in contest']
    for column in columns_to_convert:
        df[column] = df[column].str.rstrip('%').astype(float).astype('category').cat.codes
    columns_to_convert_to_codes = ['date', 'hour', 'category']
    for column in columns_to_convert_to_codes:
        df[column] = df[column].astype('category').cat.codes

    # Split the data into features and target
    X = df.drop(columns=['number of goals team2'])
    y = df['number of goals team2']

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    rfc_tuned = RandomForestClassifier(n_estimators=100, max_depth=10)
    rfc_tuned.fit(X_train, y_train)

    # Calculate feature importance
    importance_df = pd.DataFrame({"Feature_Name": X.columns, "Importance": rfc_tuned.feature_importances_})
    sorted_importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Display feature importance
    st.subheader("Feature Importance")
    st.write("This chart shows the importance of each feature in predicting the number of goals scored by Team 2.")
    chart = st.bar_chart(sorted_importance_df.set_index('Feature_Name').head(15), use_container_width=True)

    # Explanation of feature importance
    st.subheader("Interpretation of Feature Importance")
    st.write("Feature importance indicates how much each feature influences the prediction.")
    st.write("Higher importance suggests stronger influence on predicting the number of goals.")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rfc_tuned)

    # Generate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Display SHAP summary plot
    st.subheader("SHAP Values")
    st.write("SHAP values reveal the impact of each feature on individual predictions.")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=10, show=False)
    st.pyplot(fig)

    # Toggle button to switch between feature importance and SHAP values
    show_feature_importance = st.checkbox("View Feature Importance Table")
    if show_feature_importance:
        st.write(sorted_importance_df.head(15))

    # Slider to adjust number of features displayed in feature importance chart
    num_features = st.slider("Number of Features to Display", min_value=5, max_value=len(sorted_importance_df), value=10)
    st.bar_chart(sorted_importance_df.set_index('Feature_Name').head(num_features), use_container_width=True)

    # Explanation of SHAP values
    st.subheader("Interpretation of SHAP Values")
    st.write("Positive SHAP values indicate features that increase the predicted number of goals.")
    st.write("Negative SHAP values indicate features that decrease the predicted number of goals.")
    st.write("Higher magnitude suggests stronger impact on predictions.")



elif st.session_state.app_mode == 'MLflow & Deployment':
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    from mlflow import log_metric

    import mlflow
    import os

    st.subheader("MLflow & Deployment")
    st.sidebar.subheader("Play FIFA World Cup Song")
    st.sidebar.markdown("<p style='color: #ffcc00; font-family: Impact; font-size: 16px;'>🎵 Enjoy the below chosen FIFA World Cup song for you! 🎶 Feel free to adjust the volume or stop the song whenever you want. 🎧</p>", unsafe_allow_html=True)


    st.sidebar.audio(audio_fifa_4, format='audio/mp3', start_time=0)
    st.image(image_quatar2022_2, use_column_width=True)
    st.title("MLflow & Deployment")

    df = pd.read_csv('FIFAWorldCup2022.csv')
    X = df[["assists team2", "attempts inside the penalty area  team2"]]  # Features
    y = df['number of goals team2']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    dt = DecisionTreeClassifier(random_state=42)

    param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}

    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    mlflow.log_params(best_params)

    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    log_metric("accuracy", accuracy)
    log_metric("precision", precision)
    log_metric("recall", recall)
    log_metric("f1", f1)

    mlflow.sklearn.log_model(best_dt, "best")

    model_path = "best_model"
    if os.path.exists(model_path):
        try:
            import shutil
            shutil.rmtree(model_path)
        except OSError as e:
            st.error(f"An error occurred while deleting the previous model: {e}")

    mlflow.sklearn.save_model(best_dt, model_path)
    st.subheader("Performance Metrics:")
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    bars = ax.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=12)

    st.pyplot(fig)



    # Assuming metrics_values and metrics_names are defined elsewhere
    metrics_values = [25, 35, 20, 20]
    metrics_names = ['Metric A', 'Metric B', 'Metric C', 'Metric D']

    st.subheader("Additional Visualization (Pie Chart):")
    st.write("The pie chart illustrates the distribution of performance metrics.")

    # Create the pie chart with custom colors
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(metrics_values, labels=metrics_names, autopct='%1.1f%%', startangle=140, colors=['blue', 'green', 'orange', 'red'], wedgeprops=dict(width=0.4))

    # Set the color of autopct text to black for better visibility
    plt.setp(autotexts, size=12, weight="bold", color="black")

    # Adjust pie chart properties
    ax.axis('equal')
    ax.set_title('Performance Metrics Distribution')

    # Add values next to pie chart slices
    for i, text in enumerate(texts):
        text.set_text(f'{metrics_names[i]}: {metrics_values[i]:.3f}')

    # Display the pie chart
    st.pyplot(fig)


    st.info("Hover over the bars in the bar graph to view exact values. Click on the pie chart segments to see percentage breakdown.")

    st.subheader("Additional Insights:")
    st.write("Let's dive deeper into the performance metrics to understand their significance:")
    st.write("- **Accuracy**: Indicates the overall correctness of the model's predictions. A higher accuracy suggests better performance.")
    st.write("- **Precision**: Measures the correctness of positive predictions. It's the ratio of true positive predictions to all positive predictions made by the model.")
    st.write("- **Recall**: Reflects the model's ability to find all positive samples. It's the ratio of true positive predictions to all actual positive samples.")
    st.write("- **F1 Score**: Harmonic mean of precision and recall. It provides a balance between precision and recall, especially when dealing with imbalanced datasets.")

    import streamlit as st

    # Define questions and answers
    questions = [
        {
            "question": "Which country won the first ever FIFA World Cup in 1930?",
            "options": ["", "Brazil", "Uruguay", "Argentina", "Italy"],
            "answer": "Uruguay"
        },
        {
            "question": "Who is the all-time leading goal scorer in FIFA World Cup history?",
            "options": ["", "Pele", "Miroslav Klose", "Lionel Messi", "Cristiano Ronaldo"],
            "answer": "Miroslav Klose"
        }
    ]

    # Define congratulatory message
    congrats_message = "🎉 Congratulations! You got it right! 🎉"

    # Define function to display question and options
    def display_question(question_obj):
        st.subheader(question_obj["question"])
        selected_option = st.radio("Select an option:", options=question_obj["options"])
        if selected_option == question_obj["answer"] and selected_option != "":
            st.success(congrats_message)
        elif selected_option != "":
            st.warning("Oops! That's not correct. Keep trying!")

    # Prediction Page with Mindrefreshing Feature
    if st.session_state.app_mode == 'MLflow & Deployment':
        st.title("Mindrefreshing Feature: FIFA World Cup Trivia")
        st.markdown("Test your knowledge with these fun FIFA World Cup trivia questions!")

        # Display questions and options
        for i, question in enumerate(questions, 1):
            st.write(f"**Question {i}:**")
            display_question(question)

            # Option to play again for each question
            play_again = st.button("Play Again", key=f"play_again_{i}")
            if play_again:
                # Reset session state to reload questions
                st.session_state.app_mode = 'MLflow & Deployment'
       



# Conclusion Page
elif st.session_state.app_mode == 'Conclusion':
    st.subheader("Conclusion")
  

    # Play the FIFA song
    st.sidebar.subheader("Play FIFA World Cup Song")
    st.sidebar.markdown("<p style='color: #ffcc00; font-family: Impact; font-size: 16px;'>🎵 Enjoy the below chosen FIFA World Cup song for you! 🎶 Feel free to adjust the volume or stop the song whenever you want. 🎧</p>", unsafe_allow_html=True)

    st.sidebar.audio(audio_fifa, format='audio/mp3')



    st.title("FIFA World Cup 2022 Data Analysis - Conclusion 👋")
    st.video(video_concu, format='video/mp4')


    # Set page background color and font
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6; /* Light gray background */
                font-family: Arial, sans-serif; /* Choose a clean sans-serif font */
                color: #333; /* Dark gray text color */
            }
            h2 {
                color: #0047ab; /* Blue header color */
            }
            p {
                line-height: 1.5; /* Improve readability with slightly increased line spacing */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Insights about team performance
    st.markdown("## Team Performance Insights")
    st.markdown("1. **Accuracy Matters:** Teams with precise shots tend to score more goals.")
    st.markdown("2. **Seize the Opportunities:** More shots on target often translate to more scoring chances.")
    st.markdown("3. **Balancing Act:** Teams that excel in attack also need to maintain a solid defense.")
    st.markdown("4. **Team Play:** Assists play a crucial role in achieving higher goal counts.")
    st.markdown("5. **Defensive Tactics:** Aggressive defensive strategies can lead to fewer goals conceded.")

    # Limitations
    st.markdown("## Limitations")
    st.markdown("1. **Correlation, Not Causation:** While our models show strong correlations, causation cannot be definitively claimed.")
    st.markdown("2. **Room for Improvement:** Our prediction models require refinement for greater accuracy.")
    st.markdown("3. **Work in Progress:** Currently, our analysis does not predict game winners, as it wasn't our primary focus.")

    # Future directions
    st.markdown("## Future Directions")
    st.markdown("1. **Time Is Key:** Investigate the impact of specific game minutes on goal likelihood in real-time.")
    st.markdown("2. **Beyond the Numbers:** Explore sentiment analysis to understand player and fan dynamics and their influence on goals.")
    st.markdown("3. **Stay Updated:** Implement real-time data analysis for timely insights during tournaments.")
    st.markdown("4. **Enhanced Predictions:** Develop robust models based on historical data to predict match outcomes and winners.")

    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from codecarbon import EmissionsTracker
    import tensorflow as tf

    # Load FIFA World Cup dataset from CSV
    data = pd.read_csv('FIFAWorldCup2022.csv')

    # Select independent variables (features) and target variable
    selected_features = ["assists team2", "attempted defensive line breaks team2", "on target attempts team2", "inbehind offers to receive team2", "possession team2", "passes completed team2"]
    selected_target = 'number of goals team2'
    # Remove percentage signs and convert to float
    for column in selected_features:
        data[column] = data[column].astype(str).str.rstrip('%').astype(float)

    # Extract selected features and target variable from the dataset
    X = data[selected_features]
    y = data[selected_target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the emissions tracker for linear regression
    tracker_linear = EmissionsTracker()
    tracker_linear.start()

    # Train the linear regression model
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)

    # Predict the house prices
    predictions_linear = model_linear.predict(X_test)

    # Stop the emissions tracker for linear regression
    emissions_linear = tracker_linear.stop()
    print(f"Estimated emissions for training the linear regression model: {emissions_linear:.4f} kg of CO2")

    # Evaluate the linear regression model
    mse_linear = mean_squared_error(y_test, predictions_linear)
    rmse_linear = np.sqrt(mse_linear)
    print("Root Mean Squared Error (Linear Regression):", rmse_linear)

    # Define a function to load MNIST dataset
    def load_mnist():
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return (x_train, y_train), (x_test, y_test)

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Initialize the emissions tracker for neural network
    tracker_nn = EmissionsTracker()
    tracker_nn.start()

    # Define and train the neural network model
    model_nn = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_nn.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    model_nn.fit(x_train, y_train, epochs=3)

    # Stop the emissions tracker for neural network
    emissions_nn = tracker_nn.stop()
    print(f"Estimated emissions for training the neural network model: {emissions_nn:.4f} kg of CO2")

    # Combine emissions from both models
    total_emissions = emissions_linear + emissions_nn

    # Calculate accuracy of the neural network model
    test_loss, test_accuracy = model_nn.evaluate(x_test, y_test, verbose=2)
    print("Test Accuracy (Neural Network):", test_accuracy)

    # Button to toggle the visibility of the output
    if st.button("Show Emissions and Model Evaluation"):

        # Estimated emissions and model evaluation
        st.markdown("## Model Evaluation and Environmental Impact")

        st.markdown("A. Estimated emissions for training the linear regression model:")
        st.write(f"{emissions_linear:.4f} kg of CO2")

        st.markdown("B. Root Mean Squared Error (Linear Regression):")
        st.write(rmse_linear)

        st.markdown("C. Estimated emissions for training the neural network model:")
        st.write(f"{emissions_nn:.4f} kg of CO2")

        st.markdown("D. Total emissions:")
        st.write(f"{total_emissions:.4f} kg of CO2")

        st.markdown("E. Test Accuracy (Neural Network):")
        st.write(test_accuracy)

    # Display questions below emissions button
    st.markdown("## Kahoot Quiz")

    questions = [
        {
            "question": "Which of the following is a key component that increases the likelihood of a team scoring goals?",
            "options": ["", "On Target Attempts", "Number of Fans in the Stadium", "Weather Conditions", "Team's Mascot"],
            "answer": "On Target Attempts",
            "selected_option": None
        },
        {
            "question": "Which factor is most crucial for a team to create scoring opportunities?",
            "options": ["", "Number of Goals Conceded", "Successful Passes Completed by the Team", "Team's Jersey Color", "Length of the Grass on the Field"],
            "answer": "Successful Passes Completed by the Team",
            "selected_option": None
        }
    ]

    congrats_message = "🎉 Congratulations! You got it right! 🎉"

    # Define function to display question and options
    def display_question(question_obj):
        st.markdown(f"### {question_obj['question']}")
        selected_option = st.radio("Select your answer:", options=question_obj["options"], key=question_obj["question"])
        if selected_option == question_obj["answer"]:
            st.success(congrats_message)
        elif selected_option and selected_option != "":
            st.warning("Oops! That's not correct. Better luck next time!")

    # Quiz
    for i, question in enumerate(questions, 1):
        st.write(f"**Question {i}:**")
        display_question(question)

    # Conclusion and Surprise Element
    st.markdown("<p style='font-family: Arial; font-size: 24px; font-weight: bold; color: white;'>## 🏆 That's a Wrap! 🏆</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Arial; font-size: 20px; color: white;'>🎉 Thanks for exploring our FIFA World Cup 2022 Data Analysis app! 🎉</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Arial; font-size: 20px; color: white;'>Hope you enjoyed discovering insights and trends in the data.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Arial; font-size: 20px; color: white;'>Congratulations on your journey through football analytics!</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: Arial; font-size: 20px; color: white;'>Here's a special surprise just for you!</p>", unsafe_allow_html=True)

    

    # Additional Shocking Feature
    st.subheader("Reveal Secret")
    if st.button("Reveal Secret"):
        st.balloons()
        st.success("🎉 You found the hidden treasure! Enjoy your victory! 🎉")
        # Advanced Feature: Continuously moving and shining balloons
        st.write('<style>div.st-balloons > img {animation: balloon-float 2s linear infinite, balloon-spin 4s linear infinite;}</style>', unsafe_allow_html=True)
        st.write('<style>@keyframes balloon-float {0% {transform: translateY(0);} 50% {transform: translateY(-20px);} 100% {transform: translateY(0);}} @keyframes balloon-spin {from {transform: rotate(0deg);} to {transform: rotate(360deg);}}</style>', unsafe_allow_html=True)

