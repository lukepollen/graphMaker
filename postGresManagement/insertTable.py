import pandas as pd
from sqlalchemy import create_engine


def insert_dataframe_to_postgres(df, table_name, conn_params):
    """
    Insert a pandas DataFrame into a PostgreSQL table.

    Parameters:
    df (pd.DataFrame): The DataFrame to be inserted
    table_name (str): The name of the table where data will be inserted
    conn_params (dict): Dictionary containing connection parameters
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(
            f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['database']}")

        # Insert DataFrame into PostgreSQL table
        df.to_sql(table_name, engine, if_exists='replace', index=False)

        print(f"DataFrame successfully inserted into {table_name} table.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'username': ['alice', 'bob'],
        'email': ['alice@example.com', 'bob@example.com']
    }
    df = pd.DataFrame(data)

    # Connection parameters
    conn_params = {
        'host': 'localhost',  # Hostname of the PostgreSQL server (Docker container)
        'port': 5432,  # Port of the PostgreSQL server. Can be a Docker container with the port exposed.
        'database': 'postgres',  # Database name
        'user': 'postgres',  # Database user
        'password': 'mysecretpassword'  # Database password
    }

    # Table name
    table_name = 'users'

    # Insert DataFrame into PostgreSQL
    insert_dataframe_to_postgres(df, table_name, conn_params)
