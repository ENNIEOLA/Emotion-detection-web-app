number = int(input("Enter a number: "))
if number % 2 != 0:
    print("It is odd")
    for num in range(8):
      print(num)
else:
    print("It is even")
    for num in range(5):
      print(num)


for i in range(101):
   print(i)
i = int(input("Enter a number bettwen 1 -100: "))
import random
def main():
   number= random.randint(1,10)
   print('The number is: ',number)
main()
heads = 1
tails = 2
tosses = 10