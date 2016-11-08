def test(x,y):
    return not (not x or not y) and (x or y) and True
print(test(True, True))
print(test(True, False))
print(test(False, True))
print(test(False, False))

number = 16
if number%2 == 0:
    print("hooray")
elif number%16 == 0:
    print("success")
else:
    print("failure")

k = 1
i = 100
while i >= 0:
    i = i-5
    print(str(k))
    k += 1

while True:
    break
