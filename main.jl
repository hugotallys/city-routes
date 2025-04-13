using JSON
using DataStructures

struct City
    name::String
    state::String
    latitude::Float64
    longitude::Float64
    population::Int64
end

struct CityNetwork
    cities::Vector{City}
    adjacency_list::Vector{Vector{Int}}
    threshold_km::Float64
end

function load_cities(filepath::String)::Vector{City}
    # Read and parse the JSON file
    data = open(filepath, "r") do file
        JSON.parse(read(file, String))
    end
    
    # Create a vector to store city objects
    cities = Vector{City}()
    
    # Extract only the required properties and convert to appropriate types
    for city_data in data
        name = city_data["city"]
        state = city_data["state"]
        latitude = city_data["latitude"]
        longitude = city_data["longitude"]
        
        # Convert population from string to integer
        population = parse(Int64, city_data["population"])
        
        # Create a new City object and push to the vector
        push!(cities, City(name, state, latitude, longitude, population))
    end
    
    return cities
end

function haversine_distance(lat1::Float64, lon1::Float64, lat2::Float64, lon2::Float64)::Float64
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = deg2rad(lat1)
    lon1_rad = deg2rad(lon1)
    lat2_rad = deg2rad(lat2)
    lon2_rad = deg2rad(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance  # Returns distance in kilometers
end

function distance_between_cities(city1::City, city2::City)::Float64
    return haversine_distance(
        city1.latitude, city1.longitude, 
        city2.latitude, city2.longitude
    )
end

function create_distance_matrix(cities::Vector{City})::Matrix{Float64}
    n = length(cities)
    distances = zeros(Float64, n, n)
    
    # Compute distances between all pairs of cities
    for i in 1:n
        for j in (i+1):n
            dist = distance_between_cities(cities[i], cities[j])
            # Store distance in both positions (symmetric matrix)
            distances[i, j] = dist
            distances[j, i] = dist
        end
        # Distance from a city to itself is 0 (already set by zeros())
    end
    
    return distances
end

function build_city_network(cities::Vector{City}, threshold_km::Float64)::CityNetwork
    n = length(cities)
    adjacency_list = [Int[] for _ in 1:n]
    
    # Check all city pairs
    for i in 1:n
        for j in (i+1):n
            if distance_between_cities(cities[i], cities[j]) <= threshold_km
                push!(adjacency_list[i], j)
                push!(adjacency_list[j], i)  # Undirected graph
            end
        end
    end
    
    return CityNetwork(cities, adjacency_list, threshold_km)
end

function get_connected_cities(network::CityNetwork, city_index::Int)::Vector{Int}
    return network.adjacency_list[city_index]
end

mutable struct Node
    state::Int
    parent::Union{Node, Nothing}
    action::Int
    path_cost::Float64
end

function AStarSearch(start::Int, goal::Int, network::CityNetwork, distances::Matrix{Float64})::Vector{Node}
    node = Node(start, nothing, -1, 0.0)
    frontier = PriorityQueue{Node, Float64}()
    explored = Set{Int}()
    
    enqueue!(frontier, node, 0.0)

    while true
        if isempty(frontier)
            return []  # Failure! No path found
        end
        
        current_node = dequeue!(frontier)

        if current_node.state == goal
            solution = Vector{Node}()
            while !isnothing(current_node.parent)
                push!(solution, current_node)
                current_node = current_node.parent
            end
            return solution
        end
        
        push!(explored, current_node.state)
        
        for neighbor in get_connected_cities(network, current_node.state)
            if !(neighbor ∈ explored || neighbor ∈ keys(frontier))
                cost = current_node.path_cost + distances[current_node.state, neighbor]
                # a-star variation add heuristic
                cost += distances[neighbor, goal]  # Add heuristic (straight-line distance)
                child_node = Node(
                    neighbor, 
                    current_node, 
                    -1,  # Action is not defined here
                    cost
                )
                enqueue!(frontier, child_node, cost)
            elseif neighbor ∈ keys(frontier)
                # Check if the new path is cheaper
                existing_node = frontier[neighbor]
                if current_node.path_cost + distances[current_node.state, neighbor] < existing_node.path_cost
                    # Update the node in the priority queue
                    child_node = Node(
                        neighbor, 
                        current_node, 
                        -1,  # Action is not defined here
                        current_node.path_cost + distances[current_node.state, neighbor] + distances[neighbor, goal]
                    )
                    enqueue!(frontier, child_node, child_node.path_cost)
                end
            end
        end
    end
end

function main()
    # Load city data
    data_path = joinpath(@__DIR__, "data", "cities.json")
    cities = load_cities(data_path)
    println("Loaded $(length(cities)) cities.")
    
    # Print some sample data to verify it works
    println("\nFirst 5 cities:")
    for i in 1:min(5, length(cities))
        city = cities[i]
        println("$(city.name): lat $(city.latitude), long $(city.longitude), population $(city.population)")
    end
    
    # Example of calculating distance between two cities
    if length(cities) >= 2
        ny = cities[1]  # New York is the first city in your list
        la = cities[2]  # Los Angeles is the second city
        dist = distance_between_cities(ny, la)
        println("\nDistance between $(ny.name) and $(la.name) is $(round(dist, digits=2)) kilometers")
    end

    # Create distance matrix
    distance_matrix = create_distance_matrix(cities)
    println("\nCreated distance matrix with size $(size(distance_matrix))")

    # Calculate memory size
    num_cities = length(cities)
    matrix_bytes = num_cities * num_cities * 8
    matrix_kb = matrix_bytes / 1024
    matrix_mb = matrix_kb / 1024
    println("Matrix memory size: $(round(matrix_bytes)) bytes ≈ $(round(matrix_kb, digits=2)) KB ≈ $(round(matrix_mb, digits=2)) MB")

    # Get the closest city to New York
    if length(cities) >= 2
        new_york_dist = distance_matrix[1, :]
        closest_city_index = argmin(new_york_dist[2:end]) + 1  # +1 to account for the offset
        closest_city = cities[closest_city_index]
        println("\nClosest city to New York is $(closest_city.name) with a distance of $(round(new_york_dist[closest_city_index], digits=2)) km")
        
        # Get the farthest city from New York
        farthest_city_index = argmax(new_york_dist[2:end]) + 1  # +1 to account for the offset
        farthest_city = cities[farthest_city_index]
        println("Farthest city from New York is $(farthest_city.name) with a distance of $(round(new_york_dist[farthest_city_index], digits=2)) km")
    end
    
    # Build city network (connections between cities within 200km)
    threshold_km = 200.0

    println("\nBuilding city network with threshold of $(threshold_km) km...")

    city_network = build_city_network(cities, threshold_km)
    
    # Example: Find cities connected to New York
    if length(cities) >= 1
        ny_connections = get_connected_cities(city_network, 1)
        println("\nCities within $(threshold_km) km of New York:")
        if isempty(ny_connections)
            println("  None")
        else
            for idx in ny_connections
                println("  $(cities[idx].name) ($(round(distance_matrix[1, idx], digits=1)) km)")
            end
        end
    end

    # Example: Get a random path from New York to another city with variable depth
    if length(cities) >= 6
        start_city_index = 1  # New York
        path_length = 15
        path = [start_city_index]
        path_distance = 0.0
        
        for _ in 1:path_length
            current_city_index = path[end]
            next_cities = get_connected_cities(city_network, current_city_index)
            if isempty(next_cities)
                break
            end
            next_city_index = rand(next_cities)
            push!(path, next_city_index)
            path_distance += distance_matrix[current_city_index, next_city_index]
        end
        
        println("\nRandom path from New York (index $(start_city_index)) with depth $(path_length):")
        
        for idx in path
            println("  $(cities[idx].name)")
        end

        println("Total distance: $(round(path_distance, digits=2)) km")
    end

    # Example: A-Star from New York (idx 1) to Kansas City (idx 37)

    start_city_index = 2  # Los Angeles
    goal_city_index = 32  # Sacramento

    println("\nPerforming A* Search from $(cities[start_city_index].name) to $(cities[goal_city_index].name)...")

    path = AStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)

    if isempty(path)
        println("No path found!")
    else
        println("Path found:")
        for node in path
            println("  $(cities[node.state].name) (cost: $(node.path_cost))")
        end
    end
end

main()