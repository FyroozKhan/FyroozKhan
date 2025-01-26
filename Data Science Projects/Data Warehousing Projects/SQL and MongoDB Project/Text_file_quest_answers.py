# This script contains all the queries for the given questions of Project_Option1: Healthcare Management System in SQL and MongoDB.
# An explanation of the answers is given, along with their execution times.
# The exeuction times of Q:1-4 and Q:9-10 have been measured using Python scripts, whereas 5-8 have been measured using built-in SQL and Mongodb tools
# Note: The 'table' names in SQL and 'collection' names in MongoDB are named differently in different queries (but the same datasets were used by everyone)

from pymongo import MongoClient
import mysql.connector
import time
from pprint import pprint        

#%%
# --- Step 1: Connect to MySQL server and MongoDB ---
# Establishing a connection to the MySQL server
dataBase = mysql.connector.connect(
    host ="localhost",    
    user ="root",         
    passwd ="",           
    database = "healthcarems_db"  # Specifies the database name to connect to
)


# Establishing a connection to MongoDB
print("\n--- MongoDB Connection ---\n")
client = MongoClient("mongodb://localhost:27017/")
db = client['HealthcareMS_db']
appointments = db["Appointments"]
patients = db["Patients"]
print("Connected to MongoDB successfully.\n")


#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Create Database, Tables and Collections ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# GOAL: Create the database, tables named: 'PatientDetails/' and 'AppointmentDetails'and collections 'Patients' and 'Appointments'.

# SQL Query Script:
# Creata a new database called 'healthcarems_db' in phpMyAdmin
# The following sql queries are written to create the tables: 'PatientDetails' and 'AppointmentDetails' in that database

'''
CREATE TABLE PatientDetails (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    age INT,
    contact_number VARCHAR(255),
    email VARCHAR(255)
);

-- Set the starting value for AUTO_INCREMENT
ALTER TABLE PatientDetails AUTO_INCREMENT = 101;


CREATE TABLE AppointmentDetails (
    appointment_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    date DATE,
    is_follow_up BOOLEAN,
    symptom1 VARCHAR(255),
    symptom2 VARCHAR(255),
    medication_name VARCHAR(255),
    dosage VARCHAR(255),
    frequency VARCHAR(255),
    dose_day1 INT,
    dose_day2 INT,
    dose_day3 INT,
    dose_day4 INT,
    dose_day5 INT,
    dose_day6 INT,
    dose_day7 INT
);

'''
# These commands create the database called 'HealthcareMS_db' and the tables named 'Patients' and 'Appointments' within it.
# After that, the csv files namely appointments_sql.csv and patients_sql.csv are imported to their respective tables in phpMyAdmin.


# MongoDB Shell Script Equivalent:
# Open Command Prompt and then type the following:
'''
mongosh
use HealthcareMS_db

db.createCollection("Patients")
db.createCollection("Appointments")
'''
print("Database and collections created successfully.\nImport the csv files to their respective collections in MongoDB Compass.")
# These commands select the 'HealthcareMS_db' database and create the collections 'Patients' and 'Appointments' within it.
# After that, the csv files namely appointments_mongo.csv and patients_mongo.csv are imported to their respective collections in MongoDB Compass.


# Preparing a cursor object to execute SQL queries
cursorObject = dataBase.cursor()

#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 1: Update the First Element in an Array Inside a Nested Object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Modify the first element of the dose_schedule array within treatment details for a specified appointment.

# SQL Query Script:
query = '''
UPDATE appointmentdetails
SET dose_day1 = 10
WHERE appointment_id = 5;
'''
# Answer: This command will update the appointment_id of 5 and set the new dose value of dose_day1 column to 10.
# Explanation: 
# The UPDATE statement modifies data in the appointmentdetails table.
# The SET clause specifies the new value for the dose_day1 column.
# The WHERE clause filters the rows to update based on the appointment_id.

# Measure the execution time in SQL
input("Press Enter to measure the execution time in SQL for Question 1...\n")
start_time = time.time()
cursorObject.execute(query)
dataBase.commit()  # Commit the transaction to save changes
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (SQL)\n")


# MongoDB Shell Script Equivalent:
result = '''
db.Appointments.updateOne(
  { _id: 5 },
  { $set: { "treatment_details.dose_schedule.0": 10 } }
);
'''
# Answer: This command will update the _id of 5 and set the new dose value of dose_day1 column to 10.
# Explanation: 
# The updateOne method updates a single document that matches the filter.
# The filter { _id: 5 } specifies the appointment by its unique ID.
# The $set operator modifies the first element (0 index) of the dose_schedule array within treatment_details.

# Measure the execution time in MongoDB
input("Press Enter to measure the execution time in MongoDB for Question 1...\n")
start_time = time.time()
result = appointments.update_one(
    { "_id": 5 },
    { "$set": { "treatment_details.dose_schedule.0": 10 } }
)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (MongoDB)\n")



#%%
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 2: Retrieve Patients with Specific Criteria ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# SQL Query Script:
query = '''
SELECT *
FROM patientdetails
WHERE (age < 20 OR age > 50)
  AND ((contact_number IS NOT NULL AND contact_number <> '') OR (email IS NOT NULL AND email <> ''));
'''
# Answer: This query selects all rows from the patientdetails table where: 
# The age of the patient is either below 20 or above 50 and has either a non-null and non-empty contact number or a non-null and non-empty email address. 
# Explanation: 
# This query selects all rows from the patientdetails table where:
# The age of the patient is either below 20 (age < 20) or above 50 (age > 50).
# The patient has either a non-null and non-empty contact number (contact_number IS NOT NULL AND contact_number <> '') 
# or a non-null and non-empty email address (email IS NOT NULL AND email <> '').

# Measure the execution time in SQL
input("Press Enter to measure the execution time in SQL for Question 2...\n")
start_time = time.time()
cursorObject.execute(query)
result = cursorObject.fetchall()
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (SQL)\n")


# MongoDB Shell Script Equivalent:
query = {
  "$and": [
    {
      "$or": [
        { "age": { "$lt": 20 } },
        { "age": { "$gt": 50 } }
      ]
    },
    {
      "$or": [
        { "contact_number": { "$ne": None } },
        { "email": { "$ne": None } }
      ]
    }
  ]
}

# Answer: This command will find and list patients who are either under 20 years old or over 50 years old, and who either have a contact number or an email address.
# Explanation: 
# The $and operator ensures that both conditions must be true for a document to be returned.
# The first $or operator specifies the age criteria: patients under 20 years old (age: { $lt: 20 }) or over 50 years old (age: { $gt: 50 }).
# The second $or operator specifies the contact criteria: either the contact_number field is not null (contact_number: { $ne: null }), or the email field is not null (email: { $ne: null }).

# Measure the execution time in MongoDB
input("Press Enter to measure the execution time in MongoDB for Question 2...\n")
start_time = time.time()
result = list(patients.find(query))
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (MongoDB)\n")



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 3: Increase Age by 1 Year for All Patients Over 60 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Increment the age of all patients over 60 by 1 year.

# SQL Query Script:
query = '''
UPDATE patientdetails
SET age = age + 1
WHERE age > 60;
'''
# Answer: This query increments the age of all patients who are over 60 years old by 1 year.
# Explanation:
# The UPDATE statement is used to modify the existing records in the patientdetails table.
# The SET clause specifies that the age column should be incremented by 1.
# The WHERE clause filters the rows to only include patients whose age is greater than 60.

# Measure the execution time in SQL
input("Press Enter to measure the execution time in SQL for Question 3...\n")
start_time = time.time()
cursorObject.execute(query)
dataBase.commit()  
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (SQL)\n")


# MongoDB Shell Script Equivalent:
query = { "age": { "$gt": 60 } }
update = { "$inc": { "age": 1 } }
# Answer: This command increments the age of all patients who are over 60 years old by 1 year.
# Explanation: 
# The updateMany method is used to update all documents that match the filter criteria.
# The filter { age: { $gt: 60 } } selects patients whose age is greater than 60.
# The $inc operator increments the age field by 1.

# Measure the execution time in MongoDB
input("Press Enter to measure the execution time in MongoDB for Question 3...\n")
start_time = time.time()
result = patients.update_many(query, update)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (MongoDB)\n")



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 4: Perform Aggregation to Count Appointments by Patient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Count the number of appointments each patient has had.

# SQL Query Script:
query = '''
SELECT patient_id, COUNT(*) AS appointment_count
FROM appointmentdetails
GROUP BY patient_id;
'''
# Answer: This query counts the number of appointments each patient has had.
# Explanation: 
# The SELECT statement retrieves the patient_id and the count of appointments (COUNT(*)) for each patient.
# The FROM clause specifies the appointmentdetails table.
# The GROUP BY clause groups the results by patient_id, so the count of appointments is calculated for each patient.

# Measure the execution time in SQL
input("Press Enter to measure the execution time in SQL for Question 4...\n")
start_time = time.time()
cursorObject.execute(query)
result = cursorObject.fetchall()
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (SQL)\n")


# MongoDB Shell Script Equivalent:
result = '''
db.Appointments.aggregate([
  {
    $group: {
      _id: "$patient_id",
      appointment_count: { $sum: 1 }
    }
  }
]);
'''
# Answer: This command counts the number of appointments each patient has had.
# Explanation: 
# The aggregate method is used to perform aggregation operations.
# The $group stage groups the documents by patient_id.
# The _id field is set to $patient_id to group by patient_id.
# The appointment_count field uses the $sum operator to count the number of documents in each group.

# Measure the execution time in MongoDB
input("Press Enter to measure the execution time in MongoDB for Question 4...\n")
start_time = time.time()
result = list(appointments.aggregate([
    {
        "$group": {
            "_id": "$patient_id",
            "appointment_count": { "$sum": 1 }
        }
    }
]))
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (MongoDB)\n")



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 5: Find Appointments with Symptoms Containing 'fever' but Not 'cough' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Identify and retrieve appointments where the symptoms include "fever" but exclude "cough".

# SQL Query Script:
'''
SELECT *FROM appointments_sql
WHERE ('fever' IN (symptom1, symptom2))AND NOT ('cough' IN (symptom1, symptom2));
'''
# Answer: This code produced all of the appointments where fever is present but no cough 
# Explanation: 
# SELECT, selects which dataset to use 
# WHERE, sets up the conditions for the search 
# IN and AND NOT allows to search for appointments with fever and without cough.


# MongoDB Shell Script Equivalent:
'''
db.Appointments.find({symptoms: { $in: ['fever'] },symptoms: { $nin: ['cough'] }});
'''

# Answer: The query produced all of the appointments where fever is present but not cough
# Explanation: 
# find, finds the appointments in the database appointments.
# $in and $nin allows to search for the specific conditions of with fever and without cough


# Execution Time:
# SQL : 0.0007 seconds
# MongoDB: 174ms 



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 6: Retrieve Patients with Email Addresses from Gmail ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Filter and retrieve patients whose email addresses are hosted on Gmail.

# SQL Query Script:
'''
SELECT *FROM patients_sql WHERE email LIKE '%@gmail.com';
'''
# Answer: Retrieves all of the patients that have an email that mentions gmail.com
# Explanation: 
# SELECT, selects the database that is going to be used. 
# WHERE sets up the condition that is going to be searched for 
# LIKE searches for phrases with the desired @gmail.com


# MongoDB Shell Script Equivalent:
'''
db.Patients.find({email: { $regex: /@gmail\.com$/i }});
'''

# Answer: Retrieves all of the patients that have an email that mentions gmail.com
# Explanation:
# find looks through the specific database and searches for specific conditions 
# regex looks for data with the given phrase, @gmail

# Execution Time:
# SQL : 0.0008 seconds
# MongoDB: 207 ms



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 7: List Appointments Where More Than One Symptom Was Reported ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Gather and display appointments that have recorded more than one symptom.

# SQL Query Script:
'''
SELECT *
FROM appointments_sql
WHERE symptom1 <> '' AND symptom2 <> '';
'''
# Answer: Listed all of the appointments where multiple symptoms were reported
# Explanation: 
# SELECT, FROM selects the database that is being used
# WHERE, the conditions for the query
# This query searches for the appointments where symptom1 and 2 are not empty strings 
# if they are not empty that means that there are multiple symptoms


# MongoDB Shell Script Equivalent:
'''
db.Appointments.find({$expr: { $eq: [{ $size: "$symptoms" }, 2] }});
'''

# Answer: Listed Appointments Where More Than One Symptom Was Reported
# Explanation: 
# find, looks in the specific dataset for the conditions that are trying to be met
# in this query I searched for the appointments where the array size was 2,
# since this meant there are 2 symptoms


# Execution Time:
# SQL : 0.0011 seconds
# MongoDB: 412 ms 



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 8: Remove Appointments with No Reported Symptoms~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Delete all appointment records that lack any reported symptoms to maintain data quality.

# SQL Query Script:
'''
DELETE FROM appointments_sql
WHERE symptom1 = '' AND symptom2 = '' ;
'''
# Answer: Deletes all appointments without symptoms
# Explanation: 
# DELETE FROM, selects from which database to delete the data from
# WHERE, the conditions required for deletion
# this query searches for when both symptoms are empty strings, which means 
# there are no symptoms


# MongoDB Shell Script Equivalent:
'''
db.Appointments.deleteMany({symptoms: { $size: 0 }});
'''

# Answer: Deletes all appointments without symptoms
# Explanation: 
# delteMany, deletes all the data that fit a certain condition
# in this query the appointments with a size of 0 for symptoms 
# got deleted since this meant they had no symptoms.


# Execution Time:
# SQL : 0.0063 seconds
# MongoDB: 704 ms 



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 9: Retrieve Distinct Appointment Dates ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Collect and display a list of unique appointment dates from the appointments collection.

# SQL Query Script:
query = '''
SELECT DISTINCT date 
FROM appointmentdetails;
'''
# Answer: Retrieved a list of unique appointment dates from the database.
# Explanation:
# SELECT: Specifies which columns to retrieve, in this case, the date column.
# DISTINCT: Ensures that only unique values from the selected column are returned.
# FROM: Indicates the table to query, which is appointmentdetails.

# Measure the execution time in SQL
input("Press Enter to measure the execution time in SQL for Question 9...\n")
start_time = time.time()
cursorObject.execute(query)
result = cursorObject.fetchall()
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (SQL)\n")


# MongoDB Shell Script Equivalent:
result = '''
db.Appointments.distinct("date");
'''
# Answer: Retrieved a list of unique appointment dates from the MongoDB collection.
# Explanation:
# db.appointments: Refers to the appointments collection in the current database.
# distinct("date"): Extracts unique values from the date field in the specified collection.

# Measure the execution time in MongoDB
input("Press Enter to measure the execution time in MongoDB for Question 9...\n")
start_time = time.time()
result = appointments.distinct("date")
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (MongoDB)\n")



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 10: Delete Patients Over Age 60 with No Email ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Delete Patients Over Age 60 with No Email

# SQL Query Script:
query = '''
DELETE FROM patientdetails
WHERE age > 60 AND (email IS NULL OR email = '');
'''
# Answer: Deleted patients over 60 years old with no email address from the database
# Explanation:
# DELETE FROM patientdetails: Specifies the patientdetails table from which the data will be removed.
# WHERE age > 60: Ensures the deletion applies only to records where the age of the patient is greater than 60.
# AND (email IS NULL OR email = ''): Adds a condition that the email field must either be NULL (not provided) or an empty string ('').

# Measure the execution time in SQL
input("Press Enter to measure the execution time in SQL for Question 10...\n")
start_time = time.time()
cursorObject.execute(query)
dataBase.commit()  # Commit the transaction to save changes
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (SQL)\n")


# MongoDB Shell Script Equivalent:
result = '''
db.Patients.deleteMany({
    "age": { "$gt": 60 },
    "$or": [
        { "email": { "$exists": False } },
        { "email": "" }
    ]
})
'''
# Answer: Deleted patients over 60 years old with no email address from the MongoDB collection.
# Explanation:
# db.Patients.deleteMany(): Specifies the patients collection and removes all documents matching the criteria.
# age: { "$gt": 60 }: Targets documents where the age field is greater than 60.
# $or: [...]: Combines conditions where either:
# email: { "$exists": False }: The email field does not exist in the document.
# email: "": The email field is an empty string.

# Measure the execution time in MongoDB
input("Press Enter to measure the execution time in MongoDB for Question 10...\n")
start_time = time.time()
result = patients.delete_many({
    "age": { "$gt": 60 },
    "$or": [
        { "email": { "$exists": False } },
        { "email": "" }
    ]
})
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.3f} seconds (MongoDB)\n")

# Close the cursor and the connection
cursorObject.close()
dataBase.close()
client.close()



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 11: Retrieve Patients Aged Between 30 and 50 with Contact Information ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Get patient details, including contact information, for individuals aged between 30 and 50.

# MongoDB Shell Script:
'''
db.patients_csv.find({
    "age": { "$gte": 30, "$lte": 50 },
    "contact_number": { "$exists": True, "$ne": "" }
}, { "name": 1, "age": 1, "contact_number": 1, "email": 1 })
'''
#Answer: Retrieved patient details, including contact information, for individuals aged between 30 and 50
#Explanation:
#db.patients_csv.find: This is the MongoDB method used to query the patients collection.
#"age": { "$gte": 30, "$lte": 50 }: Filters the patients to include only those whose age is greater than or equal to 30 and less than or equal to 50 (inclusive).
#"contact_number": { "$exists": True, "$ne": "" }: Ensures that only patients with a non-null and non-empty contact_number are selected.
#{ "name": 1, "age": 1, "contact_number": 1, "email": 1 }: Specifies the fields to be returned in the result: name, age, contact_number, and email. The value 1 indicates inclusion of these fields in the output.



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 12: List Appointments Marked as Follow-Up ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Identify and list all appointments that have been marked as follow-up for easy tracking.

# MongoDB Shell Script:
'''
db.Appointments.find({ "is_follow_up": true })
'''
#Answer: Listed all appointments that have been marked as follow-up
#Explanation:
#db.Appointments.find(): This function is used in MongoDB to query the appointments collection.
#{ "is_follow_up": true }: This filter condition specifies that the query should only return documents where the is_follow_up field is set to true, indicating that the appointment is a follow-up.



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 13: Retrieve All Patients' Names and Ages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Extract and display the names and ages of all patients.

# MongoDB Shell Script:
'''
db.patients.find({}, { name: 1, age: 1, _id: 0 });
'''
# Answer: This MongoDB query retrieves all documents in the patients collection and displays only name and age.
# Explanation:
# "name: 1" and "age: 1" specify that the name and age fields are included in the results.
# _id: 0 excludes the default _id field from the output.



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 14: Find Appointments with Symptoms in an Array ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Identify appointments where symptoms are stored in an array format.

# MongoDB Shell Script:
'''
db.appointments.find({ symptoms: { $type: "array" } });
'''
# Answer: This MongoDB query retrieves all documents in the appointments collection where the symptoms field is stored in an array format.
# Explanation:
# $type: "array" checks if the symptoms field in each document is of type "array".



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 15: Retrieve Patients with Names Starting with Specific Letters Using Regular Expressions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Find patients whose names start with certain letters using regex matching.

# MongoDB Shell Script:
'''
db.patients.find({ name: { $regex: "^[A-C]"} });
'''
# Answer: This MongoDB query retrieves all documents in the patients collection where the name field starts with specific letters (in this case, A, B, or C), using a regular expression for pattern matching.
# Explanation:
# $regex: "^[A-C]" is a regular expression that matches any name starting with A, B, or C.
# ^ indicates the start of the string.



#%%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n#~~~~~~~~~~~~~~~~~~~~~~~~~ Question 16: Update Medication Dosage for Specific Conditions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

# Objective: Change the dosage of a specific medication for all appointments that contain certain symptoms.

# MongoDB Shell Script:
'''
db.appointments.updateMany(
  { symptoms: { $in: ["fever", "cough"] }, "treatment_details.medication_name": "Ibuprofen" },
  { $set: { "treatment_details.dosage": "300mg" } }
);
'''
# Answer: This MongoDB query updates all documents in the appointments collection that meet specific conditions by changing the dosage of a given medication.
# Explanation:
# symptoms: { $in: ["fever", "cough"] } matches documents where the symptoms field contains either "fever" or "cough".
# "treatment_details.medication_name": "Ibuprofen" matches documents where the treatment_details.medication_name field is "Ibuprofen".
# $set updates the dosage field in treatment_details to "300mg".