MAX = 6
total = 0.0
print(f"This program would calculate the sum of {MAX} numbers you will enter.")
for counter in range (MAX):
    number = int(input("Enter a number:"))
    total += number
print ("The total is", total)

for col in range(6):
    print('*')
#to make it in the same row
for col in range(6):
    print('*', end='')