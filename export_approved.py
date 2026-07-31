import sqlite3
import pandas as pd
import json

def export_approved_data():
    db_path = "fraud_detection.db"
    conn = sqlite3.connect(db_path)
    
    # Query approved scans
    query = "SELECT * FROM scans WHERE status = 'APPROVED'"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    if df.empty:
        print("No approved/unflagged records found in the database.")
        return
        
    output_path = "approved_screenshots.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully exported {len(df)} approved records to {output_path}")

if __name__ == "__main__":
    export_approved_data()
