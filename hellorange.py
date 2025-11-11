#Example 1
for x in range(5):
    print("Hello World")

#Example 2
for num in range(1, 10,2):
    print(num)

#Example 3
print("Number\tSquare")
print("----------")
for i in range (1,11):
    square = i**2
    print(i, '\t',square)


#Another Example
Start_KPH = 60
End_KPH = 130

print("KPH\tMPH")
print("--------------")

for KPH in range (60, 131, 10):
   MPH = KPH * 0.6214
   print(KPH,'\t',format(MPH, '.1f'))

#Another Example
start = int(input('Enter the beginning number:'))
choose = int(input('How high should i go?'))
print("Number\tCube")
print("_______________-")
for i in range(start, choose+1):
   square = i**2
   print(i, '\t',square )

#another example
MAX = 6
total = 0.0
print(f"This program would calculate the sum of {MAX} numbers you will enter.")
for counter in range (MAX):
    number = int(input("Enter a number:"))
    total += number
print ("The total is", total)