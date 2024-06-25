import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd

# Simulation parameters
num_nodes = 90  # Number of sensor nodes
num_relay_nodes = 10  # Number of relay nodes
initial_energy_sensor = 1.0  # Initial energy for regular sensor nodes in Joules
initial_energy_relay = 2.0   # Initial energy for relay nodes in Joules
base_station = np.array([250, 500])  # Position of the base station (top center)
positions = np.random.rand(num_nodes + num_relay_nodes, 2) * 500  # Positions of sensor nodes and relay nodes

# Initialize energy levels
energy_levels = np.full(num_nodes + num_relay_nodes, initial_energy_sensor)
energy_levels[num_nodes:num_nodes + num_relay_nodes] = initial_energy_relay

# Function to calculate local density centrality
def calculate_local_density_centrality(pos, r=20):
    centrality = []
    for i in range(len(pos)):
        count = 0
        for j in range(len(pos)):
            if i != j and np.linalg.norm(pos[i] - pos[j]) <= r:
                count += 1
        centrality.append(count)
    return np.array(centrality)

# Fuzzy logic decision-making process with corrected rules and enhanced centrality
def calculate_ch_probability(res_energy, dist_bs, centrality, convergence):
    # Normalize inputs
    initial_energy = initial_energy_sensor  # Assuming initial_energy_sensor is the max possible initial energy
    res_energy_norm = 1 - (res_energy / initial_energy)  # Invert energy: higher energy, lower probability
    dist_bs_norm = 1 - (dist_bs / 500)  # Invert distance: closer to base station, higher probability
    centrality_norm = centrality / np.max(centrality)  # Normalize centrality to [0, 1]
    convergence_norm = convergence  # Convergence remains as is: higher convergence, higher probability

    # Fuzzy logic rules
    ch_prob = (res_energy_norm + dist_bs_norm + centrality_norm + convergence_norm) / 4
    return ch_prob

# Initialize BPNN for data fusion
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Define number of rounds
num_rounds = 100  # Total number of rounds for simulation

# Initialize lists to store data for plotting
fused_data_list = []
dead_nodes_list = []
alive_nodes_list = []
latency_list = []
packet_loss_list = []

# Energy constants (example values, adjust as per your scenario)
E_elec = 50e-9  # Energy to run transmitter and receiver electronics (J/bit)
E_fs = 10e-12  # Free space path loss model (J/bit/m^2)
d0 = 10  # Reference distance (m)
L_data = 1000  # Data packet size (bits)

# LEACH protocol with corrected fuzzy logic-based CH selection and enhanced centrality
cumulative_dead_nodes = 0
cumulative_alive_nodes = num_nodes + num_relay_nodes

for r in range(num_rounds):
    # Cluster Formation
    centrality = calculate_local_density_centrality(positions)
    clusters = [[] for _ in range(num_nodes + num_relay_nodes)]
    for i in range(num_nodes + num_relay_nodes):
        for j in range(num_nodes + num_relay_nodes):
            if i != j and np.linalg.norm(positions[i] - positions[j]) <= 20:
                clusters[i].append(j)

    # Cluster Head Selection Using Fuzzy Logic
    ch_probabilities = np.zeros(num_nodes + num_relay_nodes)
    for i in range(num_nodes + num_relay_nodes):
        if len(clusters[i]) > 0:
            res_energy = energy_levels[i]
            dist_bs = np.linalg.norm(positions[i] - base_station)
            conv = np.mean([np.linalg.norm(positions[i] - positions[j]) for j in clusters[i]])
            ch_probabilities[i] = calculate_ch_probability(res_energy, dist_bs, centrality[i], conv)

    # Select CHs based on probabilities
    is_ch = ch_probabilities > np.percentile(ch_probabilities, 95)
    ch_indices = np.where(is_ch)[0]

    # Calculate Minimum Spanning Tree (MST) within clusters
    for ch in ch_indices:
        member_indices = np.where(np.isin(np.arange(num_nodes + num_relay_nodes), clusters[ch]))[0]
        sub_positions = positions[member_indices]
        adjacency_matrix = np.linalg.norm(sub_positions[:, None] - sub_positions[None, :], axis=-1)
        mst = minimum_spanning_tree(adjacency_matrix)

        # Data Transmission from sensor nodes to cluster head node using MST
        for i, node in enumerate(member_indices):
            if node != ch:
                path = mst[i].tocoo()
                for start, end, weight in zip(path.row, path.col, path.data):
                    # Estimate energy consumption for data transmission
                    d = np.linalg.norm(positions[start] - positions[end])
                    energy_consumed = L_data * (E_elec + E_fs * (d / d0) ** 2)
                    energy_levels[start] -= energy_consumed

                    if energy_levels[start] <= 0:
                        energy_levels[start] = 0  # Prevent negative energy

                    print(f'Sensor Node {start} sends data to Sensor Node {end} via CH {ch} (energy: {energy_consumed:.4f} J)')

    # Data Aggregation and Transmission to Base Station
    for ch in ch_indices:
        member_indices = np.where(np.isin(np.arange(num_nodes + num_relay_nodes), clusters[ch]))[0]
        aggregated_data = np.mean(np.random.rand(len(member_indices)))  # Simulated data aggregation

        # Data Fusion Using BPNN
        fused_data = model.predict(np.array([[aggregated_data]]))

        # Simulate data transmission to base station
        d_bs = np.linalg.norm(positions[ch] - base_station)
        energy_consumed = L_data * (E_elec + E_fs * (d_bs / d0) ** 2)
        energy_levels[ch] -= energy_consumed

        if energy_levels[ch] <= 0:
            energy_levels[ch] = 0  # Prevent negative energy

        print(f'Round {r + 1}, CH {ch} sends fused data {fused_data[0][0]} to BS (energy: {energy_consumed:.4f} J)')

        # Store fused data for plotting
        fused_data_list.append(fused_data[0][0])

    # Energy Consumption for non-CH nodes (simplified)
    for i in range(num_nodes + num_relay_nodes):
        if not is_ch[i]:
            energy_levels[i] -= np.random.uniform(0.1, 0.3)  # Randomized energy consumption

            if energy_levels[i] <= 0:
                energy_levels[i] = 0  # Prevent negative energy

    # Check for dead nodes
    dead_nodes = np.sum(energy_levels <= 0)
    cumulative_dead_nodes += dead_nodes
    cumulative_alive_nodes -= dead_nodes

    # Store data for plotting
    dead_nodes_list.append(cumulative_dead_nodes)
    alive_nodes_list.append(cumulative_alive_nodes)

    # Calculate latency (example calculation)
    latency = np.mean(np.random.randint(10, 50, size=num_nodes + num_relay_nodes))  # Example latency calculation
    latency_list.append(latency)

    # Calculate packet loss (example calculation)
    packet_loss = np.random.uniform(0, 1)  # Randomized packet loss for simulation purposes
    packet_loss_list.append(packet_loss)

    print(f'Round {r + 1}, Cumulative Dead Nodes: {cumulative_dead_nodes}')

    # Replenish energy for nodes to extend their life
    if r < num_rounds - 1:  # Do not replenish energy after the last round
        energy_levels[:num_nodes] += np.random.uniform(0.1, 0.3, size=num_nodes)  # Randomized energy replenishment for sensors
        energy_levels[num_nodes:num_nodes + num_relay_nodes] += np.random.uniform(0.2, 0.5, size=num_relay_nodes)  # Randomized energy replenishment for relays

# Plotting graphs
plt.figure(figsize=(15, 10))

# Plot Fused Data
plt.subplot(2, 3, 1)
plt.plot(fused_data_list, marker='o')
plt.title('Fused Data')
plt.xlabel('Rounds')
plt.ylabel('Data Value')

# Plot Number of Dead Nodes
plt.subplot(2, 3, 2)
plt.plot(dead_nodes_list, marker='o', color='r')
plt.title('Cumulative Number of Dead Nodes')
plt.xlabel('Rounds')
plt.ylabel('Count')

# Plot Number of Alive Nodes
plt.subplot(2, 3, 3)
plt.plot(alive_nodes_list, marker='o', color='g')
plt.title('Cumulative Number of Alive Nodes')
plt.xlabel('Rounds')
plt.ylabel('Count')

# Plot Latency
plt.subplot(2, 3, 4)
plt.plot(latency_list, marker='o', color='b')
plt.title('Latency')
plt.xlabel('Rounds')
plt.ylabel('Latency (ms)')

# Plot Packet Loss
plt.subplot(2, 3, 5)
plt.plot(packet_loss_list, marker='o', color='m')
plt.title('Packet Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss Percentage')

plt.tight_layout()
plt.show()

# Calculate final number of dead and alive nodes
final_dead_nodes = dead_nodes_list[-1]
final_alive_nodes = alive_nodes_list[-1]

print(f"Final number of dead nodes: {final_dead_nodes}")
print(f"Final number of alive nodes: {final_alive_nodes}")

# Create table of values
data = {
    'Round': np.arange(1, num_rounds + 1),
    'Cumulative Dead Nodes': dead_nodes_list,
    'Cumulative Alive Nodes': alive_nodes_list,
    'Latency (ms)': latency_list,
    'Packet Loss': packet_loss_list,
    'Fused Data': fused_data_list,
}

df = pd.DataFrame(data)
print(df)
