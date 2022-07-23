import sqlite3

con = sqlite3.connect("Leukmia.db")
c = con.cursor()
c.execute('''CREATE TABLE Users(
        Email TEXT PRIMARY KEY,
        Password CHAR(50) NOT NULL,
        Username CHAR(50) NOT NULL
        );''')
con.commit()
c.execute('''CREATE TABLE Images(
        imageID INTEGER PRIMARY KEY AUTOINCREMENT,
        Email Text NOT NULL,
        image BLOB NOT NULL ,
        Result TEXT,
        Date TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (Email)
        REFERENCES Users (Email)
        );''')
con.commit()
# datetime('now') # while inserting the date # also try current_timestamp

# c.execute("DELETE FROM Users")
# con.commit()
# c.execute("SELECT * FROM Users")
# print(c.fetchall())
# con.commit()
