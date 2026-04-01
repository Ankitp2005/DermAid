import sqlite3
import csv
import os
from datetime import datetime

class CaseLogger:
    def __init__(self, db_path='dermaid_cases.db'):
        # Ensure directory structure exists if given a path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # check_same_thread=False allows FastAPI or background threads to share the connection safely if needed
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                worker_id TEXT,
                phc_name TEXT,
                timestamp TEXT,
                image_path TEXT,
                condition_code TEXT,
                condition_name TEXT,
                severity_tier TEXT,
                urgency_color TEXT,
                confidence REAL,
                referral_action TEXT,
                auto_escalated INTEGER,
                uncertainty_score REAL,
                lang TEXT,
                synced INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()
    
    def log_case(self, patient_id, worker_id, phc_name, image_path, result_dict, lang='en') -> int:
        timestamp = datetime.now().isoformat()
        
        # We extract from dict gracefully to handle slightly different schema variants of result_dict
        self.cursor.execute('''
            INSERT INTO cases (
                patient_id, worker_id, phc_name, timestamp, image_path,
                condition_code, condition_name, severity_tier, urgency_color,
                confidence, referral_action, auto_escalated, uncertainty_score, lang
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            worker_id,
            phc_name,
            image_path,
            result_dict.get('condition_code', ''),
            result_dict.get('condition', ''),
            result_dict.get('severity', ''),
            result_dict.get('urgency_color', ''),
            result_dict.get('confidence_pct', 0.0),
            result_dict.get('action_title', ''),
            1 if result_dict.get('auto_escalated', False) else 0,
            result_dict.get('max_uncertainty', 0.0),
            lang
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid

    def get_cases(self, worker_id=None, date_from=None, limit=50) -> list:
        query = "SELECT * FROM cases WHERE 1=1"
        params = []
        
        if worker_id:
            query += " AND worker_id = ?"
            params.append(worker_id)
            
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        colnames = [desc[0] for desc in self.cursor.description]
        return [dict(zip(colnames, row)) for row in rows]
        
    def get_stats(self) -> dict:
        # Total cases
        self.cursor.execute("SELECT COUNT(*) FROM cases")
        total_cases = self.cursor.fetchone()[0]
        
        # Cases today
        today_prefix = datetime.now().strftime('%Y-%m-%d')
        self.cursor.execute("SELECT COUNT(*) FROM cases WHERE timestamp LIKE ?", (f"{today_prefix}%",))
        cases_today = self.cursor.fetchone()[0]
        
        # Red alerts today
        self.cursor.execute("SELECT COUNT(*) FROM cases WHERE timestamp LIKE ? AND urgency_color = 'RED'", (f"{today_prefix}%",))
        red_alerts_today = self.cursor.fetchone()[0]
        
        # Top conditions historically 
        self.cursor.execute("SELECT condition_code, COUNT(*) as c FROM cases GROUP BY condition_code ORDER BY c DESC LIMIT 3")
        top_conditions = self.cursor.fetchall()
        
        # Referral Rate (Percentage of NON-'Low Risk' cases)
        self.cursor.execute("SELECT COUNT(*) FROM cases WHERE severity_tier != 'Low Risk'")
        referrals = self.cursor.fetchone()[0]
        referral_rate = (referrals / total_cases * 100) if total_cases > 0 else 0.0
        
        return {
            "total_cases": total_cases,
            "cases_today": cases_today,
            "red_alerts_today": red_alerts_today,
            "top_conditions": top_conditions,
            "referral_rate": round(referral_rate, 2)
        }

    def export_csv(self, output_path='dermaid_export.csv'):
        self.cursor.execute("SELECT * FROM cases")
        rows = self.cursor.fetchall()
        colnames = [desc[0] for desc in self.cursor.description]
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(colnames)
            writer.writerows(rows)
            
    def mark_synced(self, case_ids: list):
        if not case_ids:
            return
        
        query = f"UPDATE cases SET synced = 1 WHERE id IN ({','.join(['?']*len(case_ids))})"
        self.cursor.execute(query, case_ids)
        self.conn.commit()

    def close(self):
        self.conn.close()
