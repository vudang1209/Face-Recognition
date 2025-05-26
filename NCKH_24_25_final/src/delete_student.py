
from database import connect_db, delete_by_student_id

def main():
    student_id = input("Nhập mã sinh viên cần xóa: ").strip()
    if not student_id:
        print("Mã sinh viên không được để trống.")
        return

    conn = connect_db()
    try:
        delete_by_student_id(conn, student_id)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
