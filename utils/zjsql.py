import pandas as pd
import pymysql

CONNECT = pymysql.connect(host="localhost",
                          user="root",
                          password="20010323")
CURSOR = CONNECT.cursor()


def get_table(tb) -> pd.DataFrame:
    CURSOR.execute(f"desc {tb}")
    col = pd.DataFrame(CURSOR.fetchall())[0]
    col.name = tb

    CURSOR.execute(f"select * from {tb}")
    return pd.DataFrame(CURSOR.fetchall(), columns=col)
