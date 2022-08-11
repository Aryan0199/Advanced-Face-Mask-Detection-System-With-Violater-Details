import mysql.connector

try:
    connection=mysql.connector.connect(host='localhost',database='newdb',user='root',password='admin')

    sql_select_Query = "select * from employees"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    # get all records
    records = cursor.fetchall()
    print("Total number of rows in table: ", cursor.rowcount)

    print("\nPrinting each row")
    for row in records:
        n=row[1],
        if(n=='None'):
            break

        print("Id = ", row[0], )
        print("Name = ", row[1],"\n")
        names=[]
        names.append("a")
        names.append("b")
        print(names)
        names.insert(0,'')
        print(names)
        names2=['','ac','df']
        print(names2)

except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if connection.is_connected():
        connection.close()
        cursor.close()
        print("MySQL connection is closed")