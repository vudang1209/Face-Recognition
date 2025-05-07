import sqlite3

def connect_db(db_path='database/face_recognition.db'):
    conn = sqlite3.connect(db_path)
    return conn

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    ''')
    conn.commit()

def insert_encoding(conn, name, encoding):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO face_encodings (name, encoding) VALUES (?, ?)
    ''', (name, encoding))
    conn.commit()

def get_encoding(conn, name):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT encoding FROM face_encodings WHERE name = ?
    ''', (name,))
    result = cursor.fetchone()
    return result[0] if result else None

def get_all_encodings(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT name, encoding FROM face_encodings')
    return cursor.fetchall()

def delete_by_student_id(conn, student_id):
    """
    Xóa tất cả bản ghi có name bắt đầu bằng mã sinh viên (student_id)
    """
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM face_encodings WHERE name LIKE ?', (student_id + '%',))
    count = cursor.fetchone()[0]

    if count == 0:
        print(f"⚠️ Không tìm thấy dữ liệu nào với mã sinh viên: {student_id}")
        return

    cursor.execute('DELETE FROM face_encodings WHERE name LIKE ?', (student_id + '%',))
    conn.commit()
    print(f"✅ Đã xóa {count} bản ghi với mã sinh viên: {student_id}")


def close_db(conn):
    conn.close()