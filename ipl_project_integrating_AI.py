import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import openai
import json
import pprint

#Enter YOUR OpenAI key here
openai.api_key = "Your OpenAI Key"


d = pd.read_csv("ball_by_ball_ipl.csv")

#logistic regression model
logreg = LogisticRegression(max_iter=10000)

#preprocessing
df = d.drop(["Unnamed: 0", "Match ID", "Batter", "Non Striker", "Bowler", "Method", "Player Out", "Total Batter Runs",
             "Total Non Striker Runs", "Batter Balls Faced", "Batter Runs", "Non Striker Balls Faced",
             "Player Out Runs", "Player Out Balls Faced"], axis=1)
df['Innings'] = df[df['Innings'] == 2]['Innings']
df = df.dropna()

cutoff_date = "2015-04-08"

#dummy variabkes
df = pd.get_dummies(df, columns=['Venue'], dtype=int)
venues = [column for column in df if 'Venue' in column]

features = ["Runs to Get", "Balls Remaining", "Innings Wickets"]
features.extend(venues)

#training and test sets based on date
train_data = df[(df["Date"] <= "2021-10-15") & (df["Date"] > cutoff_date)]
test_data = df[df['Date'] > "2021-10-15"]

#test data for different over segments
test_data_segments = [
    test_data[(test_data['Over'] > 0) & (test_data['Over'] < 3)],
    test_data[(test_data['Over'] > 2) & (test_data['Over'] < 5)],
    test_data[(test_data['Over'] > 4) & (test_data['Over'] < 7)],
    test_data[(test_data['Over'] > 6) & (test_data['Over'] < 9)],
    test_data[(test_data['Over'] > 8) & (test_data['Over'] < 11)],
    test_data[(test_data['Over'] > 10) & (test_data['Over'] < 13)],
    test_data[(test_data['Over'] > 12) & (test_data['Over'] < 15)],
    test_data[(test_data['Over'] > 14) & (test_data['Over'] < 17)],
    test_data[(test_data['Over'] > 16) & (test_data['Over'] < 19)],
    test_data[(test_data['Over'] > 18) & (test_data['Over'] < 21)]
]

X_train = train_data[features]
Y_train = train_data["Chased Successfully"]

#scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#training the logistic regression model
logreg.fit(X_train, Y_train)

#user inputs
runs = int(input("Runs needed to win: "))
balls = int(input("Balls left in game: "))
wickets = int(input("Wickets fallen: "))
venue_input = input("Venue played in: ")

my_features = [runs, balls, wickets]

#venue features
venue_features = [0] * len(venues)
venue_key = f"Venue_{venue_input}"
if venue_key in venues:
    venue_features[venues.index(venue_key)] = 1
else:
    print(f"Venue '{venue_input}' not found. Please enter a valid venue.")

my_features.extend(venue_features)

#scaling the user input features
my_features = scaler.transform([my_features])

#prediction of winning probability
prediction = logreg.predict_proba(my_features)
win_prob = 100 * prediction[0][1]

#display the chance of winning
print("Chance of winning: " + str(win_prob) + "%")

#commentary generation using OpenAI
system_prompt = "You are Cricket [The Game] commentator, who narrates a match condition to a Cricket Lover. Your Output should follow JSON format."
user_prompt = """
Given the following Cricket match condition, and wininning probability of the Chasing team, narrate it to a form of a radio commentary making sure you have touched all the pointers provided in the match conditions and emphasis on chances of winning for the chasing team:

{match_conditions}

Your response should only contain the generated text and nothing else, and follow the following format "commentary" : Your Output Here.
"""

#match conditions formatted for commentary
match_conditions = f"""
Runs needed to win: {runs}
Balls left in game: {balls}
Wickets fallen: {wickets}
Venue played in: {venue_input}
Chance of winning: {win_prob:.2f}%
"""

def generate_cricket_commentary(match_conditions):
    user_message = user_prompt.format(match_conditions=match_conditions)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=250,
        temperature=0.7  # Adjust for creativity
    )
    return response.choices[0].message["content"].strip()

#generate and display commentary
commentary = generate_cricket_commentary(match_conditions)
print("Generated Commentary:\n", commentary)

#test segments predictions (optional)
segment_names = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5",
                 "Segment 6", "Segment 7", "Segment 8", "Segment 9", "Segment 10"]

for i in range(len(test_data_segments)):
    test_segment = test_data_segments[i]
    segment_name = segment_names[i]
    
    if not test_segment.empty:
        X_test_segment = test_segment[features]
        Y_test_segment = test_segment['Chased Successfully']
        
        X_test_segment = scaler.transform(X_test_segment)
        
        segment_predictions = logreg.predict(X_test_segment)
        segment_confusion_matrix = confusion_matrix(Y_test_segment, segment_predictions)
        
        #confusion matrix can be printed for analysis if needed
        #print(f"Confusion Matrix for {segment_name}:")
        #print(segment_confusion_matrix)








































