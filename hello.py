#ask user for their name
name = input("What's your name?").strip().title()

#Remove whitespace from str
name = name.strip()

#Capitalize user's name
name = name.title()

#Split user's name into first name and last name


first, last  = name.split(" ")
print(f"hello, {last}")
print("hello,", first)

#say hello to user
print("hello,", name)
print("hello" + "," ,name)
print(f"hello, {name}")



