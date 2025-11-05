# Clone project
git clone https://github.com/yourusername/SafeRx.git
cd SafeRx

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run backend
cd backend
uvicorn main:app --reload

# Run frontend
cd ../frontend
streamlit run app.py
