
num_students = int(input("How many students do you have? "))
num_test_scores = int(input("How many test scores per students? "))
for student in range(num_students):
    total = 0.0
    print("Student number", student +1)
    print("___________________") 
    for test_number in range(num_test_scores):
        print("Test number", test_number +1, end="" )
        scores = float(input(':'))
        total += scores
    average = total / num_test_scores
    print("The average number for student number ", student+1, 'is:', average)
    print()

    print ("These are the Fathers of Faith:")
for name in ['Haggins', 'Copeland', 'Adeboye', 'Oyedepo']:
    
    print(name, sep=",")
