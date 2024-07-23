import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained model and necessary data
model = pickle.load(open('pipe.pkl', 'rb'))
team_names = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

city_names = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Function to make predictions
def predict(batting_team, bowling_team, city, current_score, balls_left, wickets_left, last_five):
    # Calculate current run rate
    current_run_rate = current_score*6 / (120 - balls_left) if balls_left < 120 else 0.0

    # Create a DataFrame from the input data
    input_data = pd.DataFrame(data=[[batting_team, bowling_team, city, current_score, balls_left, wickets_left, current_run_rate, last_five]], 
                               columns=['batting_team', 'bowling_team', 'city', 'current_score', 'balls_left', 'wickets_left', 'crr', 'last_five'])
    
    # Make predictions using the loaded model
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Gradio Interface
iface = gr.Interface(fn=predict, 
                     inputs=[gr.inputs.Dropdown(choices=team_names, label="Batting Team"),
                             gr.inputs.Dropdown(choices=team_names, label="Bowling Team"),
                             gr.inputs.Dropdown(choices=city_names, label="City"),
                             gr.inputs.Number(label="Current Score"),
                             gr.inputs.Number(label="Balls Left"),
                             gr.inputs.Number(label="Wickets Left"),
                             gr.inputs.Number(label="Last 5 Overs Run Rate"),
                             ],
                     outputs=gr.outputs.Textbox(label="Predicted Runs"))

# Launch the Gradio interface
iface.launch()
