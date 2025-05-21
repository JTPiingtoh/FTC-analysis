numbers = [5, 4, 6] # 5

n = 0
sum = 0
average = 0

for i, number in enumerate(numbers):
    average = ((average * i) + number) / (i + 1)
    print(average)

print(average)