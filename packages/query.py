import sqlite3
import json
import pandas as pd

class Database():
    def __init__(self, data_path = './data/data.db') -> None:
        
        self.conn = sqlite3.connect(data_path)

        self.cursor = self.conn.cursor()

        self.cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS users(
                user_id INTEGER PRIMARY KEY,
                user_name TEXT
            )
            ''')

        self.cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS features(
                user_id INTEGER,
                feature TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
        
    def insert_new_user(self, user_id: int, user_name: str):
        try:
            self.cursor.execute(
                f'''
                INSERT INTO users (user_id, user_name) VALUES ({user_id}, '{user_name}')
                ''')
            self.conn.commit()
            return True
        
        except sqlite3.IntegrityError:
            return False

    def get_all_user_id(self):
        results = self.cursor.execute(
            f'''
            SELECT user_id FROM users
            ''')
        
        data = results.fetchall()
        data = [x[0] for x in data]

        return data
    
    def insert_user_feature(self, user_id: int, features: dict):
        if user_id not in self.get_all_user_id():
            return False
        
        data = {
            'features': features
        }
        
        data = json.dumps(data)

        self.cursor.execute(
            f'''
            INSERT INTO features (user_id, feature) VALUES ('{user_id}', '{data}')
            ''')
        self.conn.commit()

        return True

    def get_all_features(self, path: None | str = None):
        results = self.cursor.execute(
            f'''
            SELECT * FROM features
            ''')
        
        data = results.fetchall()

        num_of_feature = len(json.loads(data[0][1])['features'][0])
        columns = [f'feature_{i}' for i in range(0, num_of_feature)]
        columns.append('user_id')

        df = pd.DataFrame(columns=columns)

        for user_id, features_str in data:
            features_json = json.loads(features_str)

            for feature in features_json['features']:
                row = []

                row.extend(feature)
                row.append(user_id)

                df.loc[len(df.index)] = row
        if path is not None:
            df.to_excel(path, index=False)
        return df           

    def get_idx_to_name(self):
        results = self.cursor.execute(
            f'''
                SELECT * FROM users
            ''')
        
        idx_to_name = dict(results.fetchall())

        return idx_to_name
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # import os

    # os.remove('./data/data.db')
    database = Database()

    # print(database.insert_new_user(user_id= 0, user_name="A name"))
    # print(database.insert_new_user(user_id= 1, user_name="B name"))
    # print(database.insert_new_user(user_id= 2, user_name="C name"))
    # print(database.insert_new_user(user_id= 3, user_name="D name"))
    # print(database.insert_new_user(user_id= 4, user_name="E name"))

    # data = [
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5],
    # ]
    
    # print(database.insert_user_feature(user_id=0, features=data))
    # print(database.insert_user_feature(user_id=1, features=data))
    # print(database.insert_user_feature(user_id=2, features=data))

    # database.get_all_user_id()

    # print(database.get_all_features())

    # database.get_all_features('./test.xlsx')

    # a = database.get_idx_to_name()
    # print(a[0])

    database.close()
