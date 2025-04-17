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
    # Initialize data structures
    frontier = PriorityQueue{Int, Float64}() # Use city indices as keys
    visited = Dict{Int, Node}()
    explored = Set{Int}()
    g_costs = Dict{Int, Float64}() # Track actual path costs separately
    
    # Initialize starting node
    start_node = Node(start, nothing, -1, 0.0)
    visited[start] = start_node
    g_costs[start] = 0.0
    
    # Calculate f-cost (g + h) for the starting node
    f_cost = 0.0 + distances[start, goal]
    frontier[start] = f_cost

    while !isempty(frontier)
        # Get the node with lowest f-cost
        current_state, _ = dequeue_pair!(frontier)
        current_node = visited[current_state]
        
        # Check if we've reached the goal
        if current_state == goal
            return construct_path(visited, goal)
        end
        
        push!(explored, current_state)
        
        # Explore neighbors
        for neighbor in get_connected_cities(network, current_state)
            if neighbor in explored
                continue
            end
            
            # Calculate new g-cost (actual path cost)
            new_g_cost = g_costs[current_state] + distances[current_state, neighbor]
            
            if !haskey(visited, neighbor) || new_g_cost < g_costs[neighbor]
                # Update or create node
                neighbor_node = Node(neighbor, current_node, -1, new_g_cost)
                visited[neighbor] = neighbor_node
                g_costs[neighbor] = new_g_cost
                
                # Calculate f-cost (g + h)
                f_cost = new_g_cost + distances[neighbor, goal]
                frontier[neighbor] = f_cost
            end
        end
    end
    
    return Vector{Node}()  # No path found
end

function AStarSearchWithHistory(start::Int, goal::Int, network::CityNetwork, 
                               distances::Matrix{Float64}, history_file::String="search_history.txt")::Vector{Node}
    # Initialize data structures
    frontier = PriorityQueue{Int, Float64}()
    visited = Dict{Int, Node}()
    explored = Set{Int}()
    g_costs = Dict{Int, Float64}()
    
    # For recording history
    open(history_file, "w") do file
        # Write initial state
        write(file, "$start\n")
    end
    
    # History for recording expansion at each step
    expansion_history = [[start]]
    current_step = 2
    
    # Initialize starting node
    start_node = Node(start, nothing, -1, 0.0)
    visited[start] = start_node
    g_costs[start] = 0.0
    
    # Calculate f-cost for the starting node
    f_cost = distances[start, goal]
    frontier[start] = f_cost

    while !isempty(frontier)
        # Get the node with lowest f-cost
        current_state, _ = dequeue_pair!(frontier)
        current_node = visited[current_state]
        
        push!(explored, current_state)
        
        # Check if we've reached the goal
        if current_state == goal
            # Save the final state of exploration
            open(history_file, "a") do file
                write(file, join(collect(explored), ","), "\n")
            end
            
            # Add the final expansion to history
            push!(expansion_history, collect(explored))
            
            # Save all history as one file at the end
            open(history_file, "w") do file
                for step in expansion_history
                    write(file, join(step, ","), "\n")
                end
            end
            
            return construct_path(visited, goal)
        end
        
        # Track nodes that will be added to frontier in this step
        new_frontier_nodes = Int[]
        
        # Explore neighbors
        for neighbor in get_connected_cities(network, current_state)
            if neighbor in explored
                continue
            end
            
            # Calculate new g-cost (actual path cost)
            new_g_cost = g_costs[current_state] + distances[current_state, neighbor]
            
            if !haskey(visited, neighbor) || new_g_cost < g_costs[neighbor]
                # Update or create node
                neighbor_node = Node(neighbor, current_node, -1, new_g_cost)
                visited[neighbor] = neighbor_node
                g_costs[neighbor] = new_g_cost
                
                # Calculate f-cost (g + h)
                f_cost = new_g_cost + distances[neighbor, goal]
                frontier[neighbor] = f_cost
                
                # Add to the current step's frontier
                push!(new_frontier_nodes, neighbor)
            end
        end
        
        # Record the current state of exploration (all explored nodes plus new frontier nodes)
        current_expansion = vcat(collect(explored), new_frontier_nodes)
        push!(expansion_history, current_expansion)
        
        # Print current expansion step (for debugging)
        # println("Step $current_step: ", join(current_expansion, ","))
        current_step += 1
    end
    
    # Save history even if no path is found
    open(history_file, "w") do file
        for step in expansion_history
            write(file, join(step, ","), "\n")
        end        
    end
    
    return Vector{Node}()  # No path found
end

function construct_path(visited::Dict{Int, Node}, goal::Int)::Vector{Node}
    path = Vector{Node}()
    current_state = goal
    
    # Trace back from goal to start
    while haskey(visited, current_state)
        current_node = visited[current_state]
        push!(path, current_node)
        
        # Break if we've reached the start node (which has no parent)
        if current_node.parent === nothing
            break
        end
        
        # Move to parent
        current_state = current_node.parent.state
    end
    
    # Reverse the path to get start to goal order
    return reverse(path)
end

function BidirectionalAStarSearch(start::Int, goal::Int, network::CityNetwork, distances::Matrix{Float64})::Vector{Node}
    # Forward search data structures (start to goal)
    forward_frontier = PriorityQueue{Int, Float64}()
    forward_visited = Dict{Int, Node}()
    forward_explored = Set{Int}()
    
    # Initialize forward search
    start_node = Node(start, nothing, -1, 0.0)
    forward_frontier[start] = 0.0
    forward_visited[start] = start_node
    
    # Backward search data structures (goal to start)
    backward_frontier = PriorityQueue{Int, Float64}()
    backward_visited = Dict{Int, Node}()
    backward_explored = Set{Int}()
    
    # Initialize backward search
    goal_node = Node(goal, nothing, -1, 0.0)
    backward_frontier[goal] = 0.0
    backward_visited[goal] = goal_node
    
    # Best meeting point tracking
    best_cost = Inf
    best_meeting_point = -1
    
    # Continue while both frontiers have nodes
    while !isempty(forward_frontier) && !isempty(backward_frontier)
        # Check if we can terminate early with optimal path
        if best_cost < Inf && 
           peek(forward_frontier)[2] + peek(backward_frontier)[2] >= best_cost
            break
        end
        
        # Forward search expansion
        current_state, _ = dequeue_pair!(forward_frontier)
        push!(forward_explored, current_state)
        current_node = forward_visited[current_state]
        
        # Check if we've found a meeting point
        if current_state in backward_explored
            path_cost = current_node.path_cost + backward_visited[current_state].path_cost
            if path_cost < best_cost
                best_cost = path_cost
                best_meeting_point = current_state
            end
        end
        
        # Expand forward neighbors
        for neighbor in get_connected_cities(network, current_state)
            if neighbor in forward_explored
                continue
            end
            
            new_cost = current_node.path_cost + distances[current_state, neighbor]
            
            if !haskey(forward_visited, neighbor) || new_cost < forward_visited[neighbor].path_cost
                # Create or update node
                neighbor_node = Node(neighbor, current_node, -1, new_cost)
                forward_visited[neighbor] = neighbor_node
                # Priority is f(n) = g(n) + h(n) where h(n) is heuristic
                forward_frontier[neighbor] = new_cost + distances[neighbor, goal]
                
                # Check if this creates a meeting point
                if neighbor in backward_explored
                    path_cost = new_cost + backward_visited[neighbor].path_cost
                    if path_cost < best_cost
                        best_cost = path_cost
                        best_meeting_point = neighbor
                    end
                end
            end
        end
        
        # Backward search expansion
        current_state, _ = dequeue_pair!(backward_frontier)
        push!(backward_explored, current_state)
        current_node = backward_visited[current_state]
        
        # Check if we've found a meeting point
        if current_state in forward_explored
            path_cost = forward_visited[current_state].path_cost + current_node.path_cost
            if path_cost < best_cost
                best_cost = path_cost
                best_meeting_point = current_state
            end
        end
        
        # Expand backward neighbors
        for neighbor in get_connected_cities(network, current_state)
            if neighbor in backward_explored
                continue
            end
            
            new_cost = current_node.path_cost + distances[current_state, neighbor]
            
            if !haskey(backward_visited, neighbor) || new_cost < backward_visited[neighbor].path_cost
                # Create or update node
                neighbor_node = Node(neighbor, current_node, -1, new_cost)
                backward_visited[neighbor] = neighbor_node
                # Priority uses distance to start as heuristic
                backward_frontier[neighbor] = new_cost + distances[neighbor, start]
                
                # Check if this creates a meeting point
                if neighbor in forward_explored
                    path_cost = forward_visited[neighbor].path_cost + new_cost
                    if path_cost < best_cost
                        best_cost = path_cost
                        best_meeting_point = neighbor
                    end
                end
            end
        end
    end
    
    # If we found a meeting point, construct the path
    if best_meeting_point != -1
        return construct_bidirectional_path(best_meeting_point, forward_visited, backward_visited, distances)
    end
    
    return Vector{Node}()  # No path found
end

function BidirectionalAStarSearchWithHistory(start::Int, goal::Int, network::CityNetwork, 
                                           distances::Matrix{Float64}, history_file::String="bi_astar_expansion.txt")::Vector{Node}
    # Forward search data structures (start to goal)
    forward_frontier = PriorityQueue{Int, Float64}()
    forward_visited = Dict{Int, Node}()
    forward_explored = Set{Int}()
    
    # Backward search data structures (goal to start)
    backward_frontier = PriorityQueue{Int, Float64}()
    backward_visited = Dict{Int, Node}()
    backward_explored = Set{Int}()
    
    # For recording history - track expansion of both frontiers
    expansion_history = []
    
    # Initialize forward search
    start_node = Node(start, nothing, -1, 0.0)
    forward_frontier[start] = 0.0
    forward_visited[start] = start_node
    
    # Initialize backward search
    goal_node = Node(goal, nothing, -1, 0.0)
    backward_frontier[goal] = 0.0
    backward_visited[goal] = goal_node
    
    # Best meeting point tracking
    best_cost = Inf
    best_meeting_point = -1
    
    # Initialize history with start and goal
    combined_frontier = Set{Int}([start, goal])
    push!(expansion_history, collect(combined_frontier))
    
    # Continue while both frontiers have nodes
    while !isempty(forward_frontier) && !isempty(backward_frontier)
        # Check if we can terminate early with optimal path
        if best_cost < Inf && 
           peek(forward_frontier)[2] + peek(backward_frontier)[2] >= best_cost
            break
        end
        
        # Forward search expansion
        current_state, _ = dequeue_pair!(forward_frontier)
        push!(forward_explored, current_state)
        current_node = forward_visited[current_state]
        
        # Check if we've found a meeting point
        if current_state in backward_explored
            path_cost = current_node.path_cost + backward_visited[current_state].path_cost
            if path_cost < best_cost
                best_cost = path_cost
                best_meeting_point = current_state
            end
        end
        
        # New frontier nodes from forward expansion
        new_forward_frontier = Set{Int}()
        
        # Expand forward neighbors
        for neighbor in get_connected_cities(network, current_state)
            if neighbor in forward_explored
                continue
            end
            
            new_cost = current_node.path_cost + distances[current_state, neighbor]
            
            if !haskey(forward_visited, neighbor) || new_cost < forward_visited[neighbor].path_cost
                # Create or update node
                neighbor_node = Node(neighbor, current_node, -1, new_cost)
                forward_visited[neighbor] = neighbor_node
                # Priority is f(n) = g(n) + h(n) where h(n) is heuristic
                forward_frontier[neighbor] = new_cost + distances[neighbor, goal]
                
                # Add to new frontier nodes
                push!(new_forward_frontier, neighbor)
                
                # Check if this creates a meeting point
                if neighbor in backward_explored
                    path_cost = new_cost + backward_visited[neighbor].path_cost
                    if path_cost < best_cost
                        best_cost = path_cost
                        best_meeting_point = neighbor
                    end
                end
            end
        end
        
        # Backward search expansion
        current_state, _ = dequeue_pair!(backward_frontier)
        push!(backward_explored, current_state)
        current_node = backward_visited[current_state]
        
        # Check if we've found a meeting point
        if current_state in forward_explored
            path_cost = forward_visited[current_state].path_cost + current_node.path_cost
            if path_cost < best_cost
                best_cost = path_cost
                best_meeting_point = current_state
            end
        end
        
        # New frontier nodes from backward expansion
        new_backward_frontier = Set{Int}()
        
        # Expand backward neighbors
        for neighbor in get_connected_cities(network, current_state)
            if neighbor in backward_explored
                continue
            end
            
            new_cost = current_node.path_cost + distances[current_state, neighbor]
            
            if !haskey(backward_visited, neighbor) || new_cost < backward_visited[neighbor].path_cost
                # Create or update node
                neighbor_node = Node(neighbor, current_node, -1, new_cost)
                backward_visited[neighbor] = neighbor_node
                # Priority uses distance to start as heuristic
                backward_frontier[neighbor] = new_cost + distances[neighbor, start]
                
                # Add to new frontier nodes
                push!(new_backward_frontier, neighbor)
                
                # Check if this creates a meeting point
                if neighbor in forward_explored
                    path_cost = forward_visited[neighbor].path_cost + new_cost
                    if path_cost < best_cost
                        best_cost = path_cost
                        best_meeting_point = neighbor
                    end
                end
            end
        end
        
        # Create combined frontier for visualization
        combined_explored = union(forward_explored, backward_explored)
        combined_frontier = union(
            Set(keys(forward_frontier)), 
            Set(keys(backward_frontier)),
            new_forward_frontier,
            new_backward_frontier,
            combined_explored
        )
        
        # Record expansion history
        push!(expansion_history, collect(combined_frontier))
    end
    
    # Save history to file
    open(history_file, "w") do file
        for step in expansion_history
            write(file, join(step, ","), "\n")
        end
    end
    
    # If we found a meeting point, construct the path
    if best_meeting_point != -1
        solution = construct_bidirectional_path(best_meeting_point, forward_visited, backward_visited, distances)
        
        # Save solution path separately
        solution_file = replace(history_file, "expansion.txt" => "solution.txt")
        save_solution_path(solution, solution_file)
        
        return solution
    end
    
    return Vector{Node}()  # No path found
end

function construct_bidirectional_path(meeting_point::Int, forward_visited::Dict{Int, Node}, backward_visited::Dict{Int, Node}, distances::Matrix{Float64})::Vector{Node}
    # Build forward path (start to meeting point)
    forward_path = Node[]
    current = meeting_point
    
    # Add the meeting point node if it's not already included
    if haskey(forward_visited, meeting_point)
        pushfirst!(forward_path, forward_visited[meeting_point])
    end
    
    # Trace back from meeting point to start
    while haskey(forward_visited, current) && forward_visited[current].parent !== nothing
        current = forward_visited[current].parent.state
        if haskey(forward_visited, current)
            pushfirst!(forward_path, forward_visited[current])
        end
    end
    
    # Make sure the start city is included
    # Find the initial city (the one with no parent)
    for (state, node) in forward_visited
        if node.parent === nothing
            if length(forward_path) == 0 || forward_path[1].state != state
                pushfirst!(forward_path, node)
            end
            break
        end
    end
    
    # Get the meeting node for backward path construction
    meeting_node = forward_visited[meeting_point]
    
    # Build backward path (meeting point to goal)
    backward_path = Node[]
    current = meeting_point
    last_node = meeting_node
    cumulative_cost = meeting_node.path_cost
    
    # Trace forward from meeting point to goal using backward search's parent pointers
    while haskey(backward_visited, current) && backward_visited[current].parent !== nothing
        next_state = backward_visited[current].parent.state
        next_cost = cumulative_cost + distances[current, next_state]
        
        # Create a new node with forward direction
        new_node = Node(next_state, last_node, -1, next_cost)
        push!(backward_path, new_node)
        
        last_node = new_node
        cumulative_cost = next_cost
        current = next_state
    end
    
    # Combine paths and return
    return vcat(forward_path, backward_path)
end

function save_solution_path(path::Vector{Node}, filename::String="astar_solution.txt")
    # Extract the city indices from the path nodes
    indices = [node.state for node in path]
    
    # Save to file
    open(filename, "w") do file
        # Write all indices on a single line, comma-separated
        write(file, join(indices, ","))
    end
    
    println("Solution path saved to $filename")
end

function main()
    # Load city data
    data_path = joinpath(@__DIR__, "data", "cities.json")
    cities = load_cities(data_path)
    println("Loaded $(length(cities)) cities.")
    
   
    # Create distance matrix
    distance_matrix = create_distance_matrix(cities)
    println("\nCreated distance matrix with size $(size(distance_matrix))")

    # Calculate memory size
    num_cities = length(cities)
    matrix_bytes = num_cities * num_cities * 8
    matrix_kb = matrix_bytes / 1024
    matrix_mb = matrix_kb / 1024
    println("Matrix memory size: $(round(matrix_bytes)) bytes ≈ $(round(matrix_kb, digits=2)) KB ≈ $(round(matrix_mb, digits=2)) MB")

    # Build city network (connections between cities within 200km)
    threshold_km = 300.0
    println("\nBuilding city network with threshold of $(threshold_km) km...")
    city_network = build_city_network(cities, threshold_km)
    
    # Path finding algorithm comparison
    println("\n============= SEARCH ALGORITHM COMPARISON =============")
    
    # Test Case 1: Medium distance path
    start_city_index = 1  # Los Angeles
    goal_city_index = 2 # Sacramento
    
    println("\nSearching for path from $(cities[start_city_index].name) to $(cities[goal_city_index].name)")
    
    # Bidirectional A* Search with timing and memory profiling
    println("\n2. Bidirectional A* Search:")
    # Run once to compile
    BidirectionalAStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)
    
    bi_a_star_time = @elapsed begin
        bi_a_star_path = BidirectionalAStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)
    end
    bi_a_star_memory = @allocated BidirectionalAStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)
    
    # Print Bidirectional A* results
    if isempty(bi_a_star_path)
        println("  No path found!")
    else
        println("  Path found with $(length(bi_a_star_path)) steps")
        println("  Execution time: $(round(bi_a_star_time * 1000, digits=2)) ms")
        println("  Path Cost: $(round(bi_a_star_path[end].path_cost, digits=2)) km")
        println("  Memory allocated: $(round(bi_a_star_memory / 1024, digits=2)) KB")
        
        if !isempty(bi_a_star_path)
            total_cost = bi_a_star_path[end].path_cost
            println("  Total path cost: $(round(total_cost, digits=2)) km")
            
            println("  Path: ")
            for node in bi_a_star_path  # Remove the reverse() call
                println("    $(cities[node.state].name)")
            end
        end
    end
    # A* Search with timing and memory profiling
    println("\n1. A* Search:")
    # Run once to compile
    AStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)
    
    a_star_time = @elapsed begin
        a_star_path = AStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)
    end
    a_star_memory = @allocated AStarSearch(start_city_index, goal_city_index, city_network, distance_matrix)
    
    # Replace the previous call to AStarSearch with:
    history_file = joinpath(@__DIR__, "data", "astar_expansion.txt")
    a_star_path = AStarSearchWithHistory(start_city_index, goal_city_index, city_network, distance_matrix, history_file)

    # After finding a successful path:
    if !isempty(a_star_path)
        solution_file = joinpath(@__DIR__, "data", "astar_solution.txt")
        save_solution_path(a_star_path, solution_file)
    end

    # Print A* results
    if isempty(a_star_path)
        println("  No path found!")
    else
        println("  Path found with $(length(a_star_path)) steps")
        println("  Execution time: $(round(a_star_time * 1000, digits=2)) ms")
        println("  Path Cost: $(round(a_star_path[end].path_cost, digits=2)) km")
        println("  Memory allocated: $(round(a_star_memory / 1024, digits=2)) KB")
        
        if !isempty(a_star_path)
            total_cost = a_star_path[end].path_cost
            println("  Total path cost: $(round(total_cost, digits=2)) km")
            
            println("  Path: ")
            for node in reverse(a_star_path)
                println("    $(cities[node.state].name)")
            end
        end
    end
    
    
    # Comparison summary
    println("\n3. Performance Comparison:")
    if !isempty(a_star_path) && !isempty(bi_a_star_path)
        time_diff_percent = (a_star_time - bi_a_star_time) / a_star_time * 100
        memory_diff_percent = (a_star_memory - bi_a_star_memory) / a_star_memory * 100
        
        println("  Time difference: Bidirectional A* is $(round(abs(time_diff_percent), digits=2))% $(time_diff_percent > 0 ? "faster" : "slower")")
        println("  Memory difference: Bidirectional A* uses $(round(abs(memory_diff_percent), digits=2))% $(memory_diff_percent > 0 ? "less" : "more") memory")
    end
    
    # Test Case 2: Longer distance path
    println("\n\n============= CHALLENGING PATH COMPARISON =============")
    
    # Try a more difficult path - New York to Los Angeles
    far_start_city_index = 1   # New York
    far_goal_city_index = 2   # Los Angeles (or another distant city)
    
    println("\nSearching for path from $(cities[far_start_city_index].name) to $(cities[far_goal_city_index].name)")
    
    # A* Search for difficult path
    println("\n1. A* Search:")
    far_a_star_time = @elapsed begin
        far_a_star_path = AStarSearch(far_start_city_index, far_goal_city_index, city_network, distance_matrix)
    end
    far_a_star_memory = @allocated AStarSearch(far_start_city_index, far_goal_city_index, city_network, distance_matrix)
    
    # Print A* results
    if isempty(far_a_star_path)
        println("  No path found!")
    else
        println("  Path found with $(length(far_a_star_path)) steps")
        println("  Execution time: $(round(far_a_star_time * 1000, digits=2)) ms")
        println("  Path Cost: $(round(far_a_star_path[end].path_cost, digits=2)) km")
        println("  Memory allocated: $(round(far_a_star_memory / 1024, digits=2)) KB")
        
    end
    
    # Bidirectional A* Search for difficult path
    println("\n2. Bidirectional A* Search:")
    far_bi_a_star_time = @elapsed begin
        far_bi_a_star_path = BidirectionalAStarSearch(far_start_city_index, far_goal_city_index, city_network, distance_matrix)
    end
    far_bi_a_star_memory = @allocated BidirectionalAStarSearch(far_start_city_index, far_goal_city_index, city_network, distance_matrix)

    # Add this to your main() function

    # Bidirectional A* Search with history for visualization
    println("\nRunning Bidirectional A* Search with history tracking...")
    bi_history_file = joinpath(@__DIR__, "data", "bi_astar_expansion.txt")
    bi_a_star_path_with_history = BidirectionalAStarSearchWithHistory(
        far_start_city_index, 
        far_goal_city_index, 
        city_network, 
        distance_matrix, 
        bi_history_file
    )

    println("Bidirectional A* search history saved to $bi_history_file")

    # Print Bidirectional A* results
    if isempty(far_bi_a_star_path)
        println("  No path found!")
    else
        println("  Path found with $(length(far_bi_a_star_path)) steps")
        println("  Execution time: $(round(far_bi_a_star_time * 1000, digits=2)) ms")
        println("  Path Cost: $(round(far_bi_a_star_path[end].path_cost, digits=2)) km")
        println("  Memory allocated: $(round(far_bi_a_star_memory / 1024, digits=2)) KB")
      
    end
    
    # Comparison summary for challenging path
    println("\n3. Performance Comparison (Challenging Path):")
    if !isempty(far_a_star_path) && !isempty(far_bi_a_star_path)
        time_diff_percent = (far_a_star_time - far_bi_a_star_time) / far_a_star_time * 100
        memory_diff_percent = (far_a_star_memory - far_bi_a_star_memory) / far_a_star_memory * 100
        
        println("  Time difference: Bidirectional A* is $(round(abs(time_diff_percent), digits=2))% $(time_diff_percent > 0 ? "faster" : "slower")")
        println("  Memory difference: Bidirectional A* uses $(round(abs(memory_diff_percent), digits=2))% $(memory_diff_percent > 0 ? "less" : "more") memory")
    end
end

main()