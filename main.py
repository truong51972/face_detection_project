from packages import clear_screen
from packages import register_new_face
from packages import face_detection

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

while True:
    clear_screen.clear()
    print("---------------------------------------")
    print('0. Exit.')
    print('1. Register new face.')
    print('2. Face detection.')
    print("---------------------------------------")

    while True:
        try:
            choice = int(input('Choice: '))
            break
        except ValueError:
            print('Wrong command!')

    if choice == 0:
        clear_screen.clear()
        exit()
        
    elif choice == 1:
        register_new_face.regis()

    elif choice == 2:
        face_detection.detect()