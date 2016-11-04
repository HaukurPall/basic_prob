from random import randint
import datetime

my_name = "Todd"
# both inclusive
random_number = randint(0, 1000)
now = datetime.datetime.now().date()
year_now = now.year
month_now = now.month
day_now = now.day

# See https://docs.python.org/3.4/library/stdtypes.html#str.format
print("Hello, I am {}.".format(my_name))
name = input("What is your name? Please only give me one first name followed by your entire last name.\n")
names = name.split(" ")
first_name = names[0:1]
other_names = names[1:]

"My favorite number is {}".format(random_number)

birthday = input("It's a pleasure to meet you. What is your birthday? DD/MM/YYYY\n")
birthday_list = birthday.split("/")
days_old = day_now - int(birthday_list[0])
months_old = month_now - int(birthday_list[1])
years_old = year_now - int(birthday_list[2])

if days_old < 0:
    months_old -= 1
if months_old < 0:
    years_old -= 1
age = years_old

smoke = input("Do you smoke? (yes/no)\n")
# we consider these values as true, others are false
smoke = smoke.lower() in ("y", "yes", "true", "t", "1")

print("It was nice talking to you. Let me summarize what I learnt about you.\n"
      "Your given name is {}.\n"
      "Your surname is {}.\n"
      "You are {} years old.\n"
      "Smokers: {}".format("".join(first_name), " ".join(other_names), age, smoke))
