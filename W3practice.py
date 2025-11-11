if 5 > 2:
    print("Five is greater than two!")
# This is how to use comments
'''
this is also used to make a comment
we are on fire
casting is used to specify the data type e.g
'''

# for string to be letters it has to be in quotation mark, for numbers no quotation 
x = (39484)
y = int(6)
z = (90.00000)
print(type(x))
print(y)
print(type(z))

a,b, c = "boy", "girl", "man"
print(a)
print(b)
print(c)
 # we can unpack a list
fruits = ["apple", "mango", "pineapple"]
f, g, h = fruits  
print(f)
print(g)
print(h)


def myfunc():
  global x
  x = "fantastic"

myfunc()

print("Python is " + x)