import os
from flask import Flask, redirect, render_template, request, session, flash, url_for
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

base_dir = os.path.dirname(__file__)
disease_info = pd.read_csv(os.path.join(base_dir, 'disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(base_dir, 'supplement_info.csv'), encoding='cp1252')
model = CNN.CNN(39)
model.load_state_dict(torch.load(os.path.join(base_dir, 'plant_disease_model.pt')))
model.eval()

# Database setup
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease_name TEXT NOT NULL,
            description TEXT NOT NULL,
            prevention TEXT NOT NULL,
            supplement_name TEXT,
            supplement_image TEXT,
            supplement_link TEXT,
            image_path TEXT,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('ai_engine_page'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('auth.html', form_type='login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                        (username, email, hashed_password))
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'danger')
        finally:
            conn.close()
    
    return render_template('auth.html', form_type='signup')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home_page'))

# User profile routes
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please log in to view your profile', 'warning')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    username = request.form['username']
    email = request.form['email']
    
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET username = ?, email = ? WHERE id = ?',
                    (username, email, session['user_id']))
        conn.commit()
        session['username'] = username
        flash('Profile updated successfully!', 'success')
    except sqlite3.IntegrityError:
        flash('Username or email already exists', 'danger')
    finally:
        conn.close()
    
    return redirect(url_for('profile'))

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
        return redirect(url_for('profile'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    if not check_password_hash(user['password'], current_password):
        flash('Current password is incorrect', 'danger')
        conn.close()
        return redirect(url_for('profile'))
    
    hashed_password = generate_password_hash(new_password)
    conn.execute('UPDATE users SET password = ? WHERE id = ?',
                (hashed_password, session['user_id']))
    conn.commit()
    conn.close()
    
    flash('Password changed successfully!', 'success')
    return redirect(url_for('profile'))

# Prediction history route
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to view your history', 'warning')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY predicted_at DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

# Report route
@app.route('/report')
def report():
    conn = get_db_connection()
    
    # Get total predictions count
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    
    # Get predictions per disease (as list of dicts)
    disease_stats = conn.execute('''
        SELECT disease_name, COUNT(*) as count 
        FROM predictions 
        GROUP BY disease_name 
        ORDER BY count DESC
    ''').fetchall()
    disease_stats = [dict(row) for row in disease_stats]  # Convert to list of dicts
    
    # Get predictions over time (as list of dicts)
    time_stats = conn.execute('''
        SELECT DATE(predicted_at) as date, COUNT(*) as count 
        FROM predictions 
        GROUP BY date 
        ORDER BY date
    ''').fetchall()
    time_stats = [dict(row) for row in time_stats]  # Convert to list of dicts
    
    conn.close()
    
    return render_template('report.html', 
                         total_predictions=total_predictions,
                         disease_stats=disease_stats,
                         time_stats=time_stats)
# Existing routes with modifications for user tracking
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    if 'user_id' not in session:
        flash('Please log in to use the AI Engine', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('Flask Deployed App\\static\\uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)
        
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        # Save prediction to database
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO predictions 
            (user_id, disease_name, description, prevention, supplement_name, supplement_image, supplement_link, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['user_id'], title, description, prevent, 
            supplement_name, supplement_image_url, supplement_buy_link, file_path
        ))
        conn.commit()
        conn.close()
        
        return render_template('submit.html', 
                             title=title, 
                             desc=description, 
                             prevent=prevent, 
                             image_url=image_url, 
                             pred=pred,
                             sname=supplement_name, 
                             simage=supplement_image_url, 
                             buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                         supplement_image=list(supplement_info['supplement image']),
                         supplement_name=list(supplement_info['supplement name']), 
                         disease=list(disease_info['disease_name']), 
                         buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)