import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("burnout.csv")

# Features & target
X = data[['study_hours', 'sleep_hours', 'screen_time', 'breaks', 'workload']]
y = data['burnout']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("✅ AI Study Wellness Assistant Ready!\n")

# USER INPUT
study_hours = float(input("Study hours/day: "))
sleep_hours = float(input("Sleep hours/day: "))
screen_time = float(input("Screen time: "))
breaks = float(input("Breaks per day: "))
workload = float(input("Workload (1-10): "))

# PREDICTION
prediction = model.predict([[study_hours, sleep_hours, screen_time, breaks, workload]])
prob = model.predict_proba([[study_hours, sleep_hours, screen_time, breaks, workload]])

print("\n🔍 Burnout Risk:", prediction[0])
print("📈 Confidence:", round(max(prob[0]) * 100, 2), "%")

# PRODUCTIVITY SCORE
productivity = (study_hours * 5 + sleep_hours * 6 + breaks * 4) - (screen_time * 3 + workload * 4)
print("📊 Productivity Score:", productivity)

# RISK ALERTS
print("\n🚨 Risk Alerts:")
if sleep_hours < 5:
    print("- Severe sleep deprivation")
if workload > 9:
    print("- Overloaded schedule")
if screen_time > 8:
    print("- Excessive screen exposure")

# DAILY PLAN
print("\n📅 Suggested Daily Plan:")
if prediction[0] == "High":
    print("- Study in 2-hour blocks")
    print("- Take 30 min breaks")
    print("- Sleep at least 7 hours")
elif prediction[0] == "Medium":
    print("- Maintain routine")
    print("- Add short breaks")
else:
    print("- Continue current schedule")

# BEHAVIOR INSIGHT
print("\n🧠 Behavioral Insight:")
if sleep_hours < 6:
    print("Your productivity is reduced due to low sleep")
elif screen_time > study_hours:
    print("Screen time is affecting focus")
else:
    print("Your routine looks balanced")