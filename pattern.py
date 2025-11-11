
for col in range(6):
    print('*')
#to make it in the same row
for col in range(6):
    print('*', end='')

#another example
row = int(input("\nHow many rows: "))
col = int(input("How many columns: "))
for column in range(col):
    print('*', )
for row in range(row):
    print("*", end="")

#triangle method
print("\nThe next question ")
BASE_SIZE = 15
for row in range(BASE_SIZE):
    for column in range(row+1):
        print('*',  end="")
    print()

#STAIR STEP PATTERN
print("This is for stairstep.")
NUM_STEPS = 8
for row in range(NUM_STEPS):
    for column in range(row):
        print(' ', end="")
    print('#')

