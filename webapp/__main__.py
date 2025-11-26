"""
Entry point for running the Flask app as a module: python -m webapp
"""
import sys
import os

# Add parent directory to path for imports (needed for src imports in app.py)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import app from the app module using relative import
from .app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Terrain Sense Lab - Starting Flask Application")
    print("="*60)
    print("Server will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)

