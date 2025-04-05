using JSON

# Define a struct to hold only the required properties
struct City
    name::String
    state::String
    latitude::Float64
    longitude::Float64
    population::Int64
end

function load_cities(filepath::String)
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

function haversine_distance(lat1::Float64, lon1::Float64, lat2::Float64, lon2::Float64)
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

# Example usage with your City struct:
function distance_between_cities(city1::City, city2::City)
    return haversine_distance(
        city1.latitude, city1.longitude, 
        city2.latitude, city2.longitude
    )
end

# Call the function to load cities
cities = load_cities("cities.json")

# Print some sample data to verify it works
println("Loaded $(length(cities)) cities.")
println("First 5 cities:")
for i in 1:min(5, length(cities))
    city = cities[i]
    println("$(city.name): lat $(city.latitude), long $(city.longitude), population $(city.population)")
end
# Example of calculating distance between two cities
# Example: Calculate distance between New York and Los Angeles
ny = cities[1]  # New York is the first city in your list
la = cities[2]  # Los Angeles is the second city

dist = distance_between_cities(ny, la)
println("Distance between $(ny.name) and $(la.name) is $(round(dist, digits=2)) kilometers")

function create_distance_matrix(cities::Vector{City})
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

distance_matrix = create_distance_matrix(cities)

println("Created distance matrix with size $(size(distance_matrix))")

# Calculate memory size
num_cities = length(cities)
matrix_bytes = num_cities * num_cities * 8
matrix_kb = matrix_bytes / 1024
matrix_mb = matrix_kb / 1024

println("Matrix memory size: $(round(matrix_bytes)) bytes ≈ $(round(matrix_kb, digits=2)) KB ≈ $(round(matrix_mb, digits=2)) MB")

# Get the closest city to new York

new_york_dist = distance_matrix[1, :]
closest_city_index = argmin(new_york_dist[2:end]) + 1  # +1 to account for the offset
closest_city = cities[closest_city_index]
println("Closest city to New York is $(closest_city.name) with a distance of $(round(new_york_dist[closest_city_index], digits=2)) km")

# Get the farthest city from new York
farthest_city_index = argmax(new_york_dist[2:end]) + 1  # +1 to account for the offset
farthest_city = cities[farthest_city_index]
println("Farthest city from New York is $(farthest_city.name) with a distance of $(round(new_york_dist[farthest_city_index], digits=2)) km")