import sqlite3


def get_table_info(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    return columns


def get_foreign_keys(cursor, table_name):
    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
    fks = cursor.fetchall()
    return fks


def main(db_path, output_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get a list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    with open(output_file, 'w') as file:
        for table in tables:
            table_name = table[0]
            print(table_name)
            file.write(f"Table: {table_name}\n")

            # Get column info for each table
            columns = get_table_info(cursor, table_name)
            for col in columns:
                file.write(f"  Column: {col[1]}, Type: {col[2]}, Primary Key: {col[5]}\n")

            # Get foreign key info
            fks = get_foreign_keys(cursor, table_name)
            if fks:
                file.write("  Foreign Keys:\n")
                for fk in fks:
                    file.write(f"    Table: {fk[2]}, Column: {fk[3]}, References: {fk[4]}\n")
            file.write("\n")

    # Close the connection
    conn.close()


# Replace 'your_database.db' with your database file path
# Replace 'output.txt' with your desired output file path
main('/home/jeff/LLM_database/mimicIV.db', '/home/jeff/PycharmProjects/Medical_LLM/src/scripts/output.txt')
