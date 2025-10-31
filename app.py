# CSC3600 AI Lab Project: A* Pathfinding Web Application
# This Flask backend implements the A* search algorithm to find shortest routes in Kuala Lumpur

from flask import Flask, render_template, request, jsonify
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import folium
import math
import time

# Initialize Flask web application
app = Flask(__name__)

# Global cache to store the road network graph (avoid re-downloading)
graph_cache = None

# Geocoder for converting location names to GPS coordinates
geolocator = Nominatim(user_agent="csc3600_astar_app", timeout=10)

# Dictionary of popular Kuala Lumpur locations with pre-defined coordinates
# This solves geocoding failures and provides instant location recognition
POPULAR_LOCATIONS = {
    "universiti putra malaysia": (3.0025, 101.7081),
    "upm": (3.0025, 101.7081),
    "klcc": (3.1578, 101.7118),
    "petronas twin towers": (3.1578, 101.7118),
    "bukit bintang": (3.1478, 101.7107),
    "kl sentral": (3.1337, 101.6864),
    "mid valley": (3.1176, 101.6774),
    "pavilion kl": (3.1495, 101.7137),
    "sunway pyramid": (3.0733, 101.6069),
    "batu caves": (3.2372, 101.6840),
    "merdeka square": (3.1477, 101.6953),
    "dataran merdeka": (3.1477, 101.6953),
    "national mosque": (3.1428, 101.6914),
    "masjid negara": (3.1428, 101.6914),
    "central market": (3.1459, 101.6958),
    "chinatown": (3.1459, 101.6958),
    "kl tower": (3.1529, 101.7013),
    "menara kl": (3.1529, 101.7013),
    "titiwangsa lake": (3.1833, 101.7000),
    "ioi city mall": (2.9967, 101.7167),
    "sunway lagoon": (3.0683, 101.6069),
    "one utama": (3.1497, 101.6147),
    "the gardens mall": (3.1176, 101.6774),
    "suria klcc": (3.1578, 101.7118),
    "lot 10": (3.1478, 101.7107)
}

def get_kuala_lumpur_graph():
    """
    Load or retrieve cached road network graph for Kuala Lumpur
    
    Uses OSMnx to download real road network data from OpenStreetMap.
    The graph represents roads as edges and intersections as nodes.
    Caching prevents re-downloading on every request.
    
    Returns:
        NetworkX graph with road network data
    """
    global graph_cache
    if graph_cache is None:
        print("Loading Kuala Lumpur road network... This may take a minute.")
        # Download drivable road network for Kuala Lumpur from OpenStreetMap
        # network_type="drive" means only roads accessible by car
        graph_cache = ox.graph_from_place(
            "Kuala Lumpur, Malaysia", 
            network_type="drive"
        )
        print("Road network loaded successfully!")
    return graph_cache

def geocode_location(location_name):
    """
    Convert location name to GPS coordinates (latitude, longitude)
    
    Uses multiple fallback strategies:
    1. Check popular locations dictionary first (instant, reliable)
    2. Try Nominatim geocoding with various query formats
    3. Verify coordinates are within Kuala Lumpur bounds
    
    Args:
        location_name: String name of location (e.g., "KLCC", "UPM")
    
    Returns:
        Tuple (latitude, longitude) or None if location not found
    """
    normalized_name = location_name.lower().strip()
    
    # Strategy 1: Check popular locations first (fastest, most reliable)
    if normalized_name in POPULAR_LOCATIONS:
        print(f"Found '{location_name}' in popular locations")
        return POPULAR_LOCATIONS[normalized_name]
    
    # Strategy 2: Try geocoding with different query formats
    search_queries = [
        f"{location_name}, Kuala Lumpur, Malaysia",
        f"{location_name}, KL, Malaysia",
        f"{location_name}, Malaysia",
        location_name
    ]
    
    for query in search_queries:
        try:
            print(f"Trying to geocode: {query}")
            location = geolocator.geocode(query, exactly_one=True)
            if location:
                # Verify coordinates are within Kuala Lumpur area (rough bounds)
                # Latitude: 2.8 to 3.4, Longitude: 101.4 to 101.9
                if 2.8 <= location.latitude <= 3.4 and 101.4 <= location.longitude <= 101.9:
                    print(f"Successfully geocoded '{location_name}' to ({location.latitude}, {location.longitude})")
                    return (location.latitude, location.longitude)
            time.sleep(1)  # Rate limiting to avoid API throttling
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding attempt failed for '{query}': {e}")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"Unexpected error geocoding '{query}': {e}")
            continue
    
    print(f"Could not geocode location: {location_name}")
    return None

def heuristic(node1, node2, graph):
    """
    Heuristic function for A* algorithm (h_score)
    
    Estimates the remaining distance from current node to goal node.
    Uses Euclidean distance as an approximation of straight-line distance.
    
    This is "admissible" (never overestimates), which guarantees optimal path.
    
    Args:
        node1: Current node ID
        node2: Goal node ID
        graph: Road network graph
    
    Returns:
        Estimated distance between nodes (float)
    """
    # Get coordinates of both nodes
    x1, y1 = graph.nodes[node1]['y'], graph.nodes[node1]['x']  # y=latitude, x=longitude
    x2, y2 = graph.nodes[node2]['y'], graph.nodes[node2]['x']
    
    # Calculate Euclidean distance (straight-line approximation)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def astar_pathfinding(graph, start_node, end_node):
    """
    A* Search Algorithm Implementation
    
    Finds the shortest path between two nodes using A* algorithm.
    
    A* Formula: f(n) = g(n) + h(n)
    - g(n) = actual cost from start to node n
    - h(n) = estimated cost from node n to goal (heuristic)
    - f(n) = total estimated cost of path through node n
    
    The algorithm always explores the node with lowest f(n) first,
    making it more efficient than Dijkstra's algorithm.
    
    Args:
        graph: Road network graph (NetworkX)
        start_node: Starting node ID
        end_node: Goal node ID
    
    Returns:
        Tuple: (path, distance_km, explored_nodes, search_stats)
        - path: List of node IDs representing the optimal route
        - distance_km: Total distance in kilometers
        - explored_nodes: List of nodes explored during search (for visualization)
        - search_stats: Dictionary with algorithm performance metrics
    """
    # Open set: nodes to be evaluated (priority queue based on f_score)
    # Format: [(f_score, node_id), ...]
    open_set = [(0, start_node)]
    
    # Came_from: tracks the best path - maps each node to its predecessor
    came_from = {}
    
    # g_score: actual cost from start to each node
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start_node] = 0  # Cost to reach start is 0
    
    # f_score: estimated total cost (g_score + heuristic)
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start_node] = heuristic(start_node, end_node, graph)
    
    # Track visited nodes and exploration order
    visited = set()
    explored_nodes = []  # For visualization - shows algorithm's search pattern
    nodes_evaluated = 0
    
    # Main A* loop - continues until open_set is empty or goal is found
    while open_set:
        # Sort by f_score and get node with lowest f_score (most promising)
        open_set.sort(key=lambda x: x[0])
        current_f, current = open_set.pop(0)
        
        # Track exploration for visualization
        if current not in visited:
            explored_nodes.append(current)
            nodes_evaluated += 1
        
        # Goal reached! Reconstruct and return the path
        if current == end_node:
            # Reconstruct path by following came_from backwards
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()  # Reverse to get start → end order
            
            # Calculate total distance by summing edge lengths
            total_distance = 0
            for i in range(len(path) - 1):
                edge_data = graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    edge = list(edge_data.values())[0]
                    total_distance += edge.get('length', 0)  # Length in meters
            
            distance_km = total_distance / 1000  # Convert to kilometers
            
            # Calculate search statistics for presentation
            search_stats = {
                'nodes_evaluated': nodes_evaluated,  # How many nodes A* explored
                'path_length': len(path),  # Number of intersections in optimal path
                'total_nodes': len(graph.nodes()),  # Total nodes in graph
                'efficiency_ratio': round((len(path) / nodes_evaluated) * 100, 2) if nodes_evaluated > 0 else 0
            }
            
            return path, distance_km, explored_nodes, search_stats
        
        # Mark current node as visited
        visited.add(current)
        
        # Explore all neighbors of current node
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue  # Skip already visited nodes
            
            # Get edge weight (road length in meters)
            edge_data = graph.get_edge_data(current, neighbor)
            edge = list(edge_data.values())[0]
            weight = edge.get('length', 0)
            
            # Calculate tentative g_score for this neighbor
            # g_score = cost to reach current + cost from current to neighbor
            tentative_g_score = g_score[current] + weight
            
            # If this path to neighbor is better than previous best
            if tentative_g_score < g_score[neighbor]:
                # Update best path to neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                
                # Calculate f_score = g_score + heuristic
                # This is the key to A*: prioritize nodes closer to goal
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end_node, graph)
                
                # Add neighbor to open_set if not already there
                if neighbor not in [node for _, node in open_set]:
                    open_set.append((f_score[neighbor], neighbor))
    
    # No path found (disconnected graph)
    return None, 0, [], {}

@app.route('/')
def index():
    """
    Serve the main HTML page
    
    Returns the frontend interface where users input locations
    """
    return render_template('index.html')

@app.route('/default-map', methods=['GET'])
def default_map():
    """
    API endpoint to serve default Kuala Lumpur map
    
    Displays an interactive map centered on KL with popular landmarks
    before user performs any search. Improves UX by showing the area.
    
    Returns:
        JSON with map HTML and success status
    """
    try:
        # Center coordinates for Kuala Lumpur
        kl_center = [3.1390, 101.6869]
        
        # Create Folium map centered on KL
        default_map = folium.Map(location=kl_center, zoom_start=12)
        
        # Add markers for popular locations to help users
        popular_spots = [
            ("KLCC", (3.1578, 101.7118), "blue"),
            ("Merdeka Square", (3.1477, 101.6953), "green"),
            ("KL Sentral", (3.1337, 101.6864), "orange"),
            ("Batu Caves", (3.2372, 101.6840), "purple"),
            ("Bukit Bintang", (3.1478, 101.7107), "red"),
        ]
        
        for name, coords, color in popular_spots:
            folium.Marker(
                coords,
                popup=name,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(default_map)
        
        # Convert map to HTML
        map_html = default_map._repr_html_()
        
        return jsonify({
            'success': True,
            'map_html': map_html
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/find-route', methods=['POST'])
def find_route():
    """
    API endpoint to find shortest route using A* algorithm
    
    This is the main backend endpoint that:
    1. Receives location names from frontend
    2. Converts them to GPS coordinates
    3. Downloads road network data
    4. Runs A* algorithm
    5. Generates interactive map with route
    6. Returns results as JSON
    
    Request JSON format:
        {
            "start": "Universiti Putra Malaysia",
            "end": "KLCC"
        }
    
    Response JSON format:
        {
            "success": true,
            "distance_km": 25.3,
            "map_html": "<html>...</html>",
            "nodes_explored": 1247,
            "path_length": 45,
            "efficiency_ratio": 3.6
        }
    """
    try:
        # Parse request data
        data = request.json
        start_location = data.get('start')
        end_location = data.get('end')
        
        # Validate input
        if not start_location or not end_location:
            return jsonify({'error': 'Both start and end locations are required'}), 400
        
        # Convert location names to GPS coordinates
        start_coords = geocode_location(start_location)
        end_coords = geocode_location(end_location)
        
        # Handle geocoding failures with helpful error messages
        if not start_coords:
            suggestions = list(POPULAR_LOCATIONS.keys())[:5]
            return jsonify({
                'error': f'Could not find location: {start_location}',
                'suggestions': suggestions,
                'hint': 'Try using popular locations like "KLCC", "UPM", "Bukit Bintang", etc.'
            }), 404
        if not end_coords:
            suggestions = list(POPULAR_LOCATIONS.keys())[:5]
            return jsonify({
                'error': f'Could not find location: {end_location}',
                'suggestions': suggestions,
                'hint': 'Try using popular locations like "KLCC", "UPM", "Bukit Bintang", etc.'
            }), 404
        
        # Load Kuala Lumpur road network graph
        graph = get_kuala_lumpur_graph()
        
        # Find nearest nodes on road network to our coordinates
        # OSMnx finds the closest intersection to the given GPS coordinates
        start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
        end_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
        
        # Run A* algorithm to find shortest path
        path, distance_km, explored_nodes, search_stats = astar_pathfinding(graph, start_node, end_node)
        
        # Handle case where no path exists
        if not path:
            return jsonify({'error': 'No route found between these locations'}), 404
        
        # Create interactive map centered between start and end
        map_center = [(start_coords[0] + end_coords[0]) / 2, 
                      (start_coords[1] + end_coords[1]) / 2]
        route_map = folium.Map(location=map_center, zoom_start=13)
        
        # Visualize explored nodes (light blue dots)
        # Shows how A* algorithm searched the graph
        # [::10] means show every 10th node to avoid cluttering the map
        for node in explored_nodes[::10]:
            node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
            folium.CircleMarker(
                node_coords,
                radius=2,
                color='#06b6d4',  # Cyan color
                fill=True,
                fillColor='#06b6d4',
                fillOpacity=0.3,
                weight=1,
                popup=f"Explored node"
            ).add_to(route_map)
        
        # Add start marker (green)
        folium.Marker(
            start_coords,
            popup=f"Start: {start_location}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(route_map)
        
        # Add end marker (red)
        folium.Marker(
            end_coords,
            popup=f"End: {end_location}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(route_map)
        
        # Draw optimal path as blue line
        route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in path]
        folium.PolyLine(
            route_coords,
            color='#3b82f6',  # Blue color
            weight=6,
            opacity=0.9,
            popup=f"Optimal Path: {distance_km:.2f} km"
        ).add_to(route_map)
        
        # Convert map to HTML for embedding in frontend
        map_html = route_map._repr_html_()
        
        # Return results as JSON
        return jsonify({
            'success': True,
            'distance_km': round(distance_km, 2),
            'map_html': map_html,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'nodes_explored': search_stats['nodes_evaluated'],
            'path_length': search_stats['path_length'],
            'total_nodes': search_stats['total_nodes'],
            'efficiency_ratio': search_stats['efficiency_ratio']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading road network data (this may take 1-2 minutes on first run)...")
    
    # Pre-load graph on startup to avoid delay on first request
    get_kuala_lumpur_graph()
    
    print("\nServer ready! Open http://localhost:5000 in your browser")
    
    # Start Flask development server
    # debug=True enables auto-reload on code changes
    app.run(debug=True, port=5000)
