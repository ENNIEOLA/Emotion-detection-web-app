print("Enter the property lot number or enter 0 to end ")
lot = int(input("Lot number: "))
TAX_FACTOR = 0.0065
while lot !=0:
    value = float(input("Enter the property value: "))
    tax = value *TAX_FACTOR
    print("Property tax: $", format(tax), )
    print("Enter the property lot number or enter 0 to end ")
    lot = int(input("Lot number: "))
  
