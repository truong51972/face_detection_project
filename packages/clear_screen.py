import os

def clear():
    # Kiểm tra hệ điều hành
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix-based systems
        os.system('clear')