# tuple collection is ordered,indexed, can contain different data types, unchangeable and allows duplicate
thistuple = ("apple", "banana",  "apple", "cherry")
print(thistuple)
print(len(thistuple))

#if it is one item you have to add comma at the nack if not python wont recognize it
hertuple = ("apple",)
print(type(hertuple))

#tuple() construcor
thistuple = tuple(("apple, banana", "apple", "cherry"))
print(thistuple)
print(thistuple[1])



