CampusNavi: Interactive Campus Pathfinding System

CampusNavi is an intelligent web application designed for seamless campus navigation. It leverages advanced search algorithms to find optimal routes on a campus map and integrates an AI assistant to provide information through natural language conversations.
!
✨ Features
Interactive Campus Map: An interactive map of the campus built with OpenStreetMap data, displaying key points of interest (POIs) and the road network.
Intelligent Pathfinding: Find the shortest or most efficient path between any two locations using a selection of classic search algorithms.
Real-time Route Metrics: Get instant feedback on your route, including total distance and estimated walking time.
AI Campus Assistant: An AI-powered chatbot that provides detailed information about campus locations, facilities, and hours.
Natural Language Navigation: Simply ask the AI for a route (e.g., "How do I get to the library from the food court?") and the system will automatically display the path.
Algorithm Comparison: A tool to compare the performance of different algorithms (A*, BFS, DFS, UCS) to visualize their efficiency.

🛠️ Tech Stack

Frontend/App Framework: Streamlit
Core Logic: Python
Graph Representation: NetworkX
Map Data: OpenStreetMap (OSM) via the OSMnx library
Interactive Mapping: Folium
AI Integration: Google Gemini API

🚀 How to Run Locally

Follow these steps to set up and run the application on your local machine.
Prerequisites
Python 3.8+
pip
Installation
Clone the repository to your local machine.
Navigate to the project directory:
cd CampusNavi


Install the required Python packages. It is recommended to use the uv package manager for speed, but pip will also work.
# Using uv (recommended)
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt







API Key Configuration
The AI assistant requires a Google Gemini API Key. You can obtain one from the Google AI Studio website.
There are two ways to set the key:
Streamlit Secrets: Create a .streamlit folder in your project root and add a file named secrets.toml with the following content:
GEMINI_API_KEY="YOUR_API_KEY_HERE"


Environment Variable: Set the API key as a system environment variable before running the app.
export GEMINI_API_KEY="YOUR_API_KEY_HERE"


Running the App
Once all dependencies and the API key are configured, run the application from your terminal:
streamlit run app.py


The application will automatically open in your default web browser.
📁 File Structure
app.py: The main application script that controls the UI and orchestrates interactions between the different components.
pathfinding.py: Contains the CampusPathfinder class, which handles the graph representation, all pathfinding algorithms (A*, BFS, DFS, UCS), and map visualization logic.
gemini_integration.py: Manages the connection to the Google Gemini API and the AI assistant's logic, including natural language processing and knowledge retrieval.
attached_assets/: A directory containing the map_1758707724808.osm file, which holds the raw campus map data.
