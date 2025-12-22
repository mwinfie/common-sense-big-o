def print_numbers_version_one(num):
    number = 2

    while number <= num:
        # If number is even, print it:
        if number % 2 == 0:
            print(number)
            pass
        
        number += 1

def print_numbers_version_two(num):
    number = 2

    while number <= num:
        print(number)

        # Increase number by 2, which, by definition is the next even number:
        number += 2

def linear_search(data_list, target_item):
    for index, item in enumerate(data_list):
        if item == target_item:
            return index

    return -1