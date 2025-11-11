score = int(input("Enter your score: "))
if score > 90:
    print('Your grade is A')
elif score > 80 and score <89:
    print('Your grae is B')
elif score > 70 and score <70:
    print('Your grade is C') 

for name in ['Eniola', 'Ruth', 'Busayo']:
    print(name)

for num in range(1,10,2):
    print(num)


print('Number\tSquare')
print('_____________')

for num in range(1,11):
    square = num**2
    print(num,square, sep="           ")


print('____________')
print('KPH\t MPH')
print('____________')

for KPH in range(60, 130, 10):
    MPH = KPH * 0.6214
    print(KPH, format(MPH, '.2f'), sep = "       ")


print("Number\t Square")
print('_____________')
end = int(input("How high do you want to go: "))

for num in range(1, end+1):
    square = num**2
    print(num, '\t',format(square, '.2f'))


max = 10
total = 0.0
for num in range(max):
    number = int(input("Enter a number: "))
    total +=number
print("The total is:", total)

        