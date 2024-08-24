import pandas as pd

from packages.query import Database
from packages.face_classify import Classifier_model
from packages import feature_extraction
from packages import clear_screen

def regis():
    database = Database()

    clear_screen.clear()
    while True:
        while True:
            try:
                user_id = int(input("Enter id: "))
                break
            except ValueError:
                clear_screen.clear()
                print('Wrong type!')

        user_name = input("Enter name: ")

        if database.insert_new_user(user_id=user_id, user_name=user_name):
            break
        else:
            clear_screen.clear()
            print('User id is already exist!')

    features = feature_extraction.extract()
    database.insert_user_feature(user_id=user_id, features=features)

    df = database.get_all_features()
    model = Classifier_model()
    model.train(df)
    model.save()

    database.close()
    input("Done! Press enter to continue!")

if __name__ == '__main__':
    regis()