#list is used to store multiple items in a single variable e.g
thislist = ["apple","banana", "apple" ,"cherry"]
print(thislist)
print(len(thislist))

# list items are defined ordered(order will not change), changeable and allow duplicate values and indexed
#list can be ofany data type
list1 = ["abs", "True", 34, "male"]
list2 = [1,4,5,6]

# negative indexing starts from the end. -1 refers to end, - 2 refers to second to the last
print(list1[-1]) 

#range indexing. you start from the number and you dont pick the ending number
yourlist = ["apple", "banana","cherry", "orange", "kiwi", "melon", "range"]
print(yourlist[2:5])

#start from the third item and end with the fifth
print(yourlist[:4])
print(yourlist[-4:-1])

#check if an item exists
if "apple" in yourlist:
    print("Yes, 'apple' is in the fruits list")
else:
    print("NO, It's not")


if "stationary" in yourlist:
    print("Yes, 'stationary' is in the  fruits list")
else:
    print("NO damn way")
 
 #change an item
yourlist[1] = "blackcurrant"
print(yourlist)

# you can change the second value by replacing it with two new values
yourlist[1:2] = ['pineapple', "watermelon"]
print(yourlist)

#we can insert an item without replacing existing values using insert()
list1.insert(2,"watermelom")
print(list1)

#we can add an item to the end of the list using append()
yourlist.append("orange")
print(yourlist) 

#we can add list to current list using extend
list1.extend(list2)
print(list1)

#we can also extend tuple
thistuple = ("kiwi", "papaya")
thislist.extend(thistuple)
print(thislist)

# we can remove using remove(). itremoves the first occurrence if there is a repitition
thislist.remove("apple")
print(thislist)

# pop removes the specified index if not specified it removes the last item
thislist.pop(1)
print(thislist)

# del removes spefied indexes too and deletes the list completely
del thislist[0]

list3 =["badore", "ajah", "ikorodu"]
del list3

list4 = ["Pinnochio", "Despicable me"]

# clear empties the list. list remains but no content
list4.clear()
print(list4)

# looping list using for
theirlist = ["Mary", "Martha", "Naomi", "Agatha"]
for x in theirlist:
    print(x)
for i in range(len(theirlist)):
    print(theirlist[i])

# with while
i = 0
while i <len(theirlist):
    print(theirlist[i])
    i= i + 1

# list comprehension used to create a new list based on values of an existing list

[print(x) for x in theirlist]

fruits = ["apple", "banana", "cherry", "kiwi","mango"]

newlist = []
for x in fruits:
    if "a" in x:
        newlist.append(x)
print(newlist)

#to sort(case sensitive, sorts uppercase before lower case or use key = str.lower)
theirlist.sort()
print(theirlist)

#to sort descending
theirlist.sort(reverse= True)
print(theirlist)

#to sort in a function use the keyword argument key(e.g used to sort how close they are to 50)
def myfunc(n):
    return abs(n-50)
ourlist = [100, 50, 65, 82, 23]
ourlist.sort(key = myfunc)
print(ourlist)

 #to reverse order
ourlist.reverse()
print(ourlist)  

#to copy list
yourlist = ourlist.copy()
print(yourlist)

yourlist = list(ourlist)
print(yourlist)

yourlist = ourlist[:]
print(yourlist)

#to join list
mylist = theirlist + yourlist
print(mylist)

for x in yourlist:
    theirlist.append(x)
print(theirlist)

theirlist.extend(yourlist)
print(theirlist)




# set is unordered, unchangeable(but youcan remove and add), unindexed and does not duplicaete
# dictionary is ordered, changeable and no duplicate.