import re

s1 = "The BodyGuard is the best album"

# Define the pattern to search for
pattern = r"Body"

# Use the search() function to search for the pattern in the string
result = re.search(pattern, s1)

# Check if a match was found
if result:
    print("Match found!")
else:
    print("Match not found.")


name = 'Lizz'
print(name[0:2])

var = '01234567'
print(var[::2])

print('1'+'2') 