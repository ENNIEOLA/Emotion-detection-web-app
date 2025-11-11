MAX = 5
total = 0.0

print('This program calculates the sum of')
print(MAX, 'numbers you will enter.')

for counter in range (MAX):
   number = int(input('Enter a number:'))
total = total + number

print('The total is', total)