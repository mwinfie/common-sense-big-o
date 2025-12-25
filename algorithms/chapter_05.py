def selection_sort(array):
    n = len(array)
    
    for i in range(n):
        lowest_number_index = i

        for j in range(i + 1, n):
            if array[j] < array[lowest_number_index]:
                lowest_number_index = j
        
        if lowest_number_index != i:
            temp = array[i]
            array[i] = array[lowest_number_index]
            array[lowest_number_index] = temp

    return array