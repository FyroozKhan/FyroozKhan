from faker import Faker
import pandas as pd
import random
import json

# Set the seed for reproducibility
seed_value = 12345
random.seed(seed_value)
Faker.seed(seed_value)

# Initialize Faker
fake = Faker()

# Helper function to generate email with a specific domain
def generate_email():
    return fake.email() if random.random() > 0.3 else f"{fake.user_name()}@gmail.com"

# Generate 100,000 patient records with unique IDs starting from 101
patients = []
for i in range(100000):
    patient_id = 101 + i  # Unique patient ID starting from 101
    patients.append({
        "_id": patient_id,  # _id starts at 101
        "name": fake.name(),
        "age": random.randint(1, 90),
        "contact_number": fake.phone_number() if random.random() > 0.2 else None,
        "email": generate_email() if random.random() > 0.3 else None
    })

# Save patients dataset as CSV
patients_df = pd.DataFrame(patients)
patients_df.to_csv("patients_mongo.csv", index=False)

# List of 10 common health symptoms
symptoms_list = ["fever", "cough", "headache", "fatigue", "shortness of breath", "nausea", "vomiting", "sore throat", "muscle pain", "dizziness"]

# Helper function to generate symptom data
def generate_symptom():
    choice = random.random()
    if choice < 0.2:
        return [], []
    symptom1 = random.choice(symptoms_list)
    return [symptom1], ([random.choice([s for s in symptoms_list if s != symptom1])] if choice >= 0.6 else [])

# Helper function to generate dose values based on frequency
def generate_doses(frequency):
    limits = {"1 time a day": 1, "2 times a day": 2, "3 times a day": 3}
    return [random.randint(0, limits[frequency]) for _ in range(7)]

# Generate 100,000 appointment records
appointments = []
for i in range(100000):
    symptom1, symptom2 = generate_symptom()
    frequency = random.choice(["1 time a day", "2 times a day", "3 times a day"])
    doses = generate_doses(frequency)
    treatment_details = {
        "medication_name": random.choice([
            "Ibuprofen", "Acetaminophen", "Aspirin", "Naproxen", "Amoxicillin", 
            "Azithromycin", "Ciprofloxacin", "Doxycycline", "Diphenhydramine", "Loratadine"
        ]),
        "dosage": f"{random.randint(1, 100)}mg",
        "frequency": frequency,
        "dose_schedule": doses
    }
    appointments.append({
        "_id": i + 1,
        "patient_id": 101 + i,  # Ensure the patient_id corresponds to the _id in patients.csv
        "date": fake.date_this_decade().strftime("%Y-%m-%d"),  # Ensure date is formatted as a string
        "is_follow_up": fake.boolean(),  # True or False
        "symptoms": json.dumps(symptom1 + symptom2),  # Store as JSON string
        "treatment_details": json.dumps(treatment_details)  # Store as JSON string
    })

# Save appointments dataset as CSV
pd.DataFrame(appointments).to_csv("appointments_mongo.csv", index=False)
