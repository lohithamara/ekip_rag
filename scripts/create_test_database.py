import sqlite3
from pathlib import Path

Path(
    "data/test_sql"
).mkdir(
    parents=True,
    exist_ok=True,
)

connection = sqlite3.connect(
    "data/test_sql/company.db"
)

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE employees(

    id INTEGER PRIMARY KEY,

    name TEXT,

    department TEXT,

    salary REAL
)
""")

cursor.executemany(

    """
    INSERT INTO employees
    VALUES(?,?,?,?)
    """,

    [

        (1,"Alice","HR",50000),

        (2,"Bob","Finance",62000),

        (3,"Charlie","Engineering",85000),

        (4,"David","Engineering",92000),

        (5,"Emma","Finance",70000),

    ],
)

connection.commit()

connection.close()

print("Database created.")