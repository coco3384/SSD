with open('9999998_00317_d_0000270.txt', 'r') as f:
    texts = f.readlines()
    for text in texts:
        print(text.split('\\')[0])