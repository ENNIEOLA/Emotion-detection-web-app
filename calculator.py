x = int(input("Enter a valid number: "))
if x > 100:
    y = 20
    z = 40
    print("y = ", y)
    print("z =", z)
else: 
    y = 30
    z = 45
    print("y = ", y)
    print("z= ", z)

    number = int(input("Enter a number between 1 and 10: "))
    roman_numerals = {
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII",
        8: "VIII",
        9: "IX",
        10: "X"
    }

    if 1  <= number <= 10:
        print("Roman numeral:", roman_numerals[number])
    else:
        print("ERROR: Number is out of range! Please enter a number from 1 through 10")

mass = float(input("Enter the object's mass:"))
weight = mass * 9.8
print("Weight of the object is", weight, "newtons")
if weight > 1000:
 print("Object is too heavy.")
elif weight < 10:
  print("Object is too light.")
else:
   print("Weight is within the range.")

month = int(input("Enter a month(1-12): "))
day = int(input("Enter a day(1-31):"))
year = int(input("Enter a two-digit year: "))
 
if month*day == year:
   print("The date is magic.")
else:
   print("The date is not magic")


five_naira = int(input("Enter the number of N5 notes: "))
ten_naira = int(input("Enter the number of N10 notes: "))
twenty_naira = int(input("Enter the number of N20 notes: "))
fifty_naira = int(input("Enter the number of N50 notes: "))

num_of_nairanotes = (5*five_naira)+(10 *ten_naira )+(20*twenty_naira)+(50*fifty_naira)
if num_of_nairanotes == 100:
  print("Congratulations for winning the game.")
elif num_of_nairanotes < 100:
   print("Number was less than 100")
else:
   print("Number is greater than 100")


color1 = str(input("Enter the first  primary color(blue, red, yellow)\n: "))
color2 = str(input("Enter the second primary color (red, blue, yellow): "))
   
primary_colors = ["red", "blue", "yellow"]

if color1 not in primary_colors or color2 not in primary_colors:
   print("Error: Invalid primary color entered")
elif color1==color2:
   print("You entered the same color twice. Try mixing two different colors")
else:
   if (color1 =="red" and color2 == "blue") or (color1 =="blue" and color2 == "red"):
      print("You get purple")
   
