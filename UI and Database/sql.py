from flask import Flask, request, jsonify,render_template
import sqlite3
import datetime

app = Flask(__name__)
DATABASE = 'vehicles.db'

def connect_to_database():
    return sqlite3.connect(DATABASE)

def create_tables():
    with connect_to_database() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries_exits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER NOT NULL,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id)
            )
        ''')
        conn.commit()

def fetch_entries(cursor, query, params=()):
    cursor.execute(query, params)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_vehicle_id(cursor, plate):
    cursor.execute('SELECT vehicle_id FROM vehicles WHERE license_plate = ?', (plate,))
    return cursor.fetchone()

@app.route('/recent_entries', methods=['GET'])
def recent_entries():
    with connect_to_database() as conn:
        cursor = conn.cursor()
        entries = fetch_entries(cursor, 'SELECT v.license_plate, e.entry_time, e.exit_time FROM entries_exits e JOIN vehicles v ON e.vehicle_id = v.vehicle_id ORDER BY e.entry_time DESC LIMIT 50')
    return jsonify(entries), 200

@app.route('/vehicle_entries/', methods=['GET'])
def vehicle_entries():
    plate = request.args.get('license_plate')
    if not plate:
        return jsonify({'error': 'License plate number is missing'}), 400

    with connect_to_database() as conn:
        cursor = conn.cursor()
        vehicle = get_vehicle_id(cursor, plate)
        if not vehicle:
            return jsonify({'error': 'No vehicle found with the provided license plate'}), 404
        vehicle_id = vehicle[0]
        entries = fetch_entries(cursor, 'SELECT entry_time, exit_time FROM entries_exits WHERE vehicle_id = ?', (vehicle_id,))
    if not entries:
        return jsonify({'message': 'No entries found for the vehicle with license plate: {}'.format(plate)}), 200
    return jsonify(entries), 200

@app.route('/vehicle_entry', methods=['POST', 'GET'])
def vehicle_entry():
    license_plate = request.args.get('license_plate')
    if not license_plate:
        return jsonify({'error': 'License plate number is missing'}), 400

    with connect_to_database() as conn:
        cursor = conn.cursor()
        
        # Check if there is an existing entry with a null exit_time for the given license plate
        cursor.execute('''
            SELECT id
            FROM entries_exits
            WHERE vehicle_id = (
                SELECT vehicle_id
                FROM vehicles
                WHERE license_plate = ?
            ) AND exit_time IS NULL
        ''', (license_plate,))
        
        existing_entry = cursor.fetchone()
        if existing_entry:
            return jsonify({'message': 'Entry already exists for license plate: {}'.format(license_plate)}), 200
        
        # If no existing entry found, proceed to insert new entry
        cursor.execute('''
            INSERT INTO vehicles (license_plate) 
            SELECT ? 
            WHERE NOT EXISTS (
                SELECT 1 
                FROM vehicles 
                WHERE license_plate = ?
            )
        ''', (license_plate, license_plate))
        
        cursor.execute('''
            INSERT INTO entries_exits (vehicle_id, entry_time)
            SELECT vehicle_id, ? 
            FROM vehicles 
            WHERE license_plate = ?
        ''', (datetime.datetime.now(), license_plate))
        
        conn.commit()

    return jsonify({'message': 'Vehicle entry recorded successfully'}), 201



@app.route('/vehicle_exit', methods=['POST', 'GET'])
def vehicle_exit():
    license_plate = request.args.get('license_plate')
    if not license_plate:
        return jsonify({'error': 'License plate number is missing'}), 400

    with connect_to_database() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE entries_exits 
            SET exit_time = ? 
            WHERE id = (SELECT id FROM entries_exits 
                        WHERE vehicle_id = (SELECT vehicle_id FROM vehicles WHERE license_plate = ?) 
                        AND exit_time IS NULL LIMIT 1)
            ''', (datetime.datetime.now(), license_plate))
        if cursor.rowcount == 0:
            return jsonify({'error': 'No matching entry or exit time already recorded for license plate: {}'.format(license_plate)}), 400
        conn.commit()

    return jsonify({'message': 'Vehicle exit recorded successfully for license plate: {}'.format(license_plate)}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)