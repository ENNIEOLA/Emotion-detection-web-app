def main():
    print('Enter the names of three friends.')
    name1 = input('Friend #1: ')
    name2 = input('Friend #2: ')
    name3 = input('Friend #3: ')

    line1 = infile.readline()
    line2 = infile.readline()
    line3 = infile.readline()

    infile = open('friend.txt', 'w')
    infile.write(name1 +'\n')
    infile.write(name2 + '\n')
    infile.write(name3 +'\n')

    print('The names were written to friends.txt')

   
    infile.close()


   