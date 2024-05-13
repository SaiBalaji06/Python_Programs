import mysql.connector
host='localhost'
user='root'
password='cmrtc123$'
database='college'
a=[]
connection=mysql.connector.connect(host=host,user=user,password=password,database=database)
if connection.is_connected():
    print('connected to database')
    cursor=connection.cursor()
    table_name='sample'
    col1='Name'
    type_of='varchar(10)'
    col2='Age'
    type_of2='int'
    q=f"""create table if not exists {table_name}({col1} {type_of},{col2} {type_of2});"""
    cursor.execute(q)
    col=('col1','col2')
    values=[('saibalaji',19),('charan',20),('komali',21)]
    aq=f"insert into {table_name}({col1},{col2})values({', '.join(['%s']*2)});"
    cursor.executemany(aq,values)
    connection.commit()
    cursor.execute('select * from sample;')
    for i in cursor.fetchall():
        print(i)
    cursor.close()
    
else:
    print('Error: not connected to Mysql')
connection.close()