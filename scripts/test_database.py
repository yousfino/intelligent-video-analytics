
from video_analytics.database.connection import (
    check_database_connection,
)


def main():
    database_name = check_database_connection()

    print("PostgreSQL connection successful!")
    print(f"Connected database: {database_name}")


if __name__ == "__main__":
    main()
