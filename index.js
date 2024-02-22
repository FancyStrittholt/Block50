//   Question 1


// You are a network designer entrusted with the responsibility of designing a computer network for a small office. The office consists of multiple rooms, and your goal is to connect them using the least amount of cable, ensuring that each room is connected to the network. You need to analyze the office layout, identify the rooms, and plan the most efficient way to connect them with cables. The objective is to minimize the required cable length while ensuring every room is connected to the network.

// Your task is to apply Prim's graph-based algorithm, which starts with an initial room and progressively adds neighboring rooms with the shortest cable connections. By iteratively expanding the network, you will create a minimum-cost spanning tree that connects all the rooms in the office. Take on the role of the network designer, employ Prim's algorithm, and determine the minimum cost of connecting all the rooms in the office using the provided scenario.

// Sample Input:- new Edge(0, 1, 4),   new Edge(0, 7, 8),   new Edge(1, 2, 8),   new Edge(1, 7, 11),   new Edge(2, 3, 7),   new Edge(2, 8, 2),   new Edge(2, 5, 4),   new Edge(3, 4, 9),   new Edge(3, 5, 14),   new Edge(4, 5, 10),   new Edge(5, 6, 2),   new Edge(6, 7, 1),   new Edge(6, 8, 6),   new Edge(7, 8, 7) in the format of (edge pairs, weights) with a total number of 9 vertices.

// Sample Output: Minimum cost to connect all rooms: 37

class PriorityQueue {
  constructor() {
    this.heap = [];
  }

  enqueue(value) {
    this.heap.push(value);
    let i = this.heap.length - 1;
    while (i > 0) {
      let j = Math.floor((i - 1) / 2);
      if (this.heap[i][0] >= this.heap[j][0]) {
        break;
      }
      [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
      i = j;
    }
  }

  dequeue() {
    if (this.heap.length === 0) {
      throw new Error('Queue is empty');
    }
    let i = this.heap.length - 1;
    const result = this.heap[0];
    this.heap[0] = this.heap[i];
    this.heap.pop();

    i--;
    let j = 0;
    while (true) {
      const left = j * 2 + 1;
      if (left > i) {
        break;
      }
      const right = left + 1;
      let k = left;
      if (right <= i && this.heap[right][0] < this.heap[left][0]) {
        k = right;
      }
      if (this.heap[j][0] <= this.heap[k][0]) {
        break;
      }
      [this.heap[j], this.heap[k]] = [this.heap[k], this.heap[j]];
      j = k;
    }

    return result;
  }

  get count() {
    return this.heap.length;
  }
}

function spanningTree(V, E, edges) {
  // Create an adjacency list representation of the graph
  const adj = new Array(V).fill(null).map(() => []);

  // Fill the adjacency list with edges and their weights
  for (let i = 0; i < E; i++) {
    const [u, v, wt] = edges[i];
    adj[u].push([v, wt]);
    adj[v].push([u, wt]);
  }

  // Create a priority queue to store edges with their weights
  const pq = new PriorityQueue();

  // Create a visited array to keep track of visited vertices
  const visited = new Array(V).fill(false);

  // Variable to store the result (sum of edge weights)
  let res = 0;

  // Start with vertex 0
  pq.enqueue([0, 0]);

  // Perform Prim's algorithm to find the Minimum Spanning Tree
  while (pq.count > 0) {
    const p = pq.dequeue();

    const wt = p[0]; // Weight of the edge
    const u = p[1]; // Vertex connected to the edge

    if (visited[u]) {
      continue; // Skip if the vertex is already visited
    }

    res += wt; // Add the edge weight to the result
    visited[u] = true; // Mark the vertex as visited

    // Explore the adjacent vertices
    for (const v of adj[u]) {
      // v[0] represents the vertex and v[1] represents the edge weight
      if (!visited[v[0]]) {
        pq.enqueue([v[1], v[0]]); // Add the adjacent edge to the priority queue
      }
    }
  }

  return res; // Return the sum of edge weights of the Minimum Spanning Tree
}

// Example usage
const graph = [
  [0, 1, 4],
  [0, 7, 8],
  [1, 2, 8],
  [1, 7, 11],
  [2, 3, 7],
  [2, 8, 2],
  [2, 5, 4],
  [3, 4, 9],
  [3, 5, 14],
  [4, 5, 10],
  [5, 6, 2],
  [6, 7, 1],
  [6, 8, 6],
  [7, 8, 7],
];
// Function call
console.log(spanningTree(9, 14, graph));




//   Question 2

// You are an aspiring computer scientist tasked with creating a function that can find the shortest path between two locations in a graph. The graph represents various locations and the roads connecting them, with each road having a specific distance associated with it. Your goal is to create a function called bfsShortestPath (graph, source, target) that takes in the graph, the source node (representing the traveler's current location), and the target node (representing the traveler's destination). The function should return an array representing the shortest path from the source to the target.

// The graph is represented using an adjacency list. This means that each location in the graph is a node, and the roads connecting them are represented as edges. The adjacency list stores the neighboring nodes for each node, allowing you to traverse the graph efficiently. Your task is to create a bfsShortestPath function, utilizing the Breadth-First Search (BFS) algorithm to find the shortest path from the source to the target. The function should return an array that represents the shortest path, starting from the source and ending at the target.

// Sample Input: A: ['B', 'C'],   B: ['A', 'D', 'E'],   C: ['A', 'F'],   D: ['B'],   E: ['B', 'F'],   F: ['C', 'E'], in the format of Vertices: (neighboring nodes) and source node will be A and Destination node will be F

// Sample Output: Shortest path from A to F: [ 'A', 'C', 'F' ]


class Graph {
    constructor() {
      this.neighbors = {}
    }
  
    addEdge(u, v) {
      if (!this.neighbors[u]) this.neighbors[u] = []
      this.neighbors[u].push(v)
    }
  
    bfs(start) {
      if (!this.neighbors[start] || !this.neighbors[start].length) {
        return [start]
      }
  
      var results = {"nodes": []},
          queue = this.neighbors[start],
          count = 1
  
      while(queue.length) {
        var node = queue.shift()
        if (!results[node] || !results[node].visited) {
          results[node] = {visited: true, steps: count}
          results["nodes"].push(node)
          if (this.neighbors[node]) {
            if (this.neighbors[node].length) {
              count++
              queue.push(...this.neighbors[node])
            } else {
              continue
            }
          }
        }
      }
      return results
    }
  
    shortestPath(start, end) {
      if (start == end) {
        return [start, end]
      }
  
      var queue = [start],
          visited = {},
          predecessor = {},
          tail = 0,
          path
  
      while(tail < queue.length) {
        var u = queue[tail++]
        if (!this.neighbors[u]) {
          continue
        }
  
        var neighbors = this.neighbors[u]
        for(var i = 0; i < neighbors.length; ++i) {
          var v = neighbors[i]
          if (visited[v]) {
            continue
          }
          visited[v] = true
          if (v === end) {   // Check if the path is complete.
            path = [ v ]   // If so, backtrack through the path.
            while (u !== start) {
              path.push(u)
              u = predecessor[u]
            }
            path.push(u)
            path.reverse()
            return path
          }
          predecessor[v] = u
          queue.push(v)
        }
      }
  
      return path
    }
  }
  
  var createGraph = function() {
    var g = new Graph()
    g.addEdge('A', 'B')
    g.addEdge('A', 'C')
  
    g.addEdge('B', 'A')
    g.addEdge('B', 'D')
    g.addEdge('B', 'E')
  
    g.addEdge('C', 'A')
    g.addEdge('C', 'F')
  
    g.addEdge('D', 'B')
  
    g.addEdge('E', 'B')
    g.addEdge('E', 'F')
  
    g.addEdge('F', 'C')
    g.addEdge('F', 'E')
  
    return g
  }
  
  
  var shortestPathGraph = createGraph()
  var path = shortestPathGraph.shortestPath('A', 'F')
  console.log(`Shortest path from A to F: ${path.join(',')}`)


//   Question 3

//   You are a cab driver in Boston, and you receive a request to pick up a passenger from a specific location. Your task is to find all possible routes to reach the passenger's location using the Depth First Search (DFS) algorithm in JavaScript. You need to implement the Depth First Search algorithm to find all possible routes from your current location (the starting node) to the passenger's location (the target node). Your goal is to provide a list of all possible routes. Implement the dfsAllRoutes(graph, source, target) function in JavaScript that takes the graph, the source node (your current location), and the target node (the passenger's location) as input. The function should return an array of all possible routes from the source to the target.

// Sample Input:  A: ["B", "C"],   B: ["A", "D", "E"],   C: ["A", "F"],   D: ["B"],   E: ["B", "F"],   F: ["C", "E"],  in the format of Vertices: (neighboring nodes) and source node will be A and Destination node will be F.

// Sample Output: All possible routes from A to F: [ [ 'A', 'B', 'E', 'F' ], [ 'A', 'C', 'F' ] ]


class Graph {
    constructor() {
        this.vertices = [];
        this.adjacent = {};
        this.edges = 0;
    }

    addVertex(v) {
        this.vertices.push(v);
        this.adjacent[v] = [];
    }

    addEdge(v, w) {
        this.adjacent[v].push(w);
        this.adjacent[w].push(v);
        this.edges++;
    }

    bfs(goal, root = this.vertices[0]) {
        let adj = this.adjacent;

        const queue = [];
        queue.push(root);

        const discovered = [];
        discovered[root] = true;

        while(queue.length) {
            let v = queue.shift();
            console.log(v);

            if (v === goal) {
                return true;
            }

            for (let i = 0; i < adj[v].length; i++) {
                if (!discovered[adj[v][i]]) {
                    discovered[adj[v][i]] = true;
                    queue.push(adj[v][i]);
                }
            }
        }

        return false;
    }
}

const g = new Graph();

g.addVertex("A");
g.addVertex("B");
g.addVertex("C");
g.addVertex("D");
g.addVertex("E");
g.addVertex("F");

g.addEdge("A","B");
g.addEdge("B","D");
g.addEdge("A","C");
g.addEdge("C","F");
g.addEdge("B","E");
g.addEdge("E","F");

console.log()



// Question 4
// Imagine you are developing a navigation system for a delivery robot that needs to navigate through a city to deliver packages efficiently. The city is represented as a graph, where each point is a location, and the edges between points represent the routes that the robot can take. Each edge has a weight associated with it, representing the distance or time required to travel from one point to another. The goal is to use Dijkstra's algorithm in JavaScript to calculate the shortest path for the robot, optimizing package delivery.

// In this scenario, the graph representing the city is as follows:

// Point A connects to Point B with a weight of 5.

// Point A connects to Point C with a weight of 2.

// Point B connects to Point D with a weight of 4.

// Point B connects to Point E with a weight of 2.

// Point C connects to Point B with a weight of 8.

// Point C connects to Point E with a weight of 7.

// Point D connects to Point E with a weight of 6.

// Point D connects to Point F with a weight of 3.

// Point E connects to Point F with a weight of 1.

 

// Sample Input:  A: { B: 5, C: 2 },   B: { D: 4, E: 2 },   C: { B: 8, E: 7 },   D: { E: 6, F: 3 },   E: { F: 1 },   F: {}, const startNode = "A"; const endNode = "F";

// Sample Output: Shortest path: A -> B -> E -> F and Distance: 8


class Node {
    constructor(val, priority) {
      this.val = val;
      this.priority = priority;
    }
  }
  
  class PriorityQueue {
    constructor() {
      this.values = [];
    }
    enqueue(val, priority) {
      let newNode = new Node(val, priority);
      this.values.push(newNode);
      this.bubbleUp();
    }
    bubbleUp() {
      let idx = this.values.length - 1;
      const element = this.values[idx];
      while (idx > 0) {
        let parentIdx = Math.floor((idx - 1) / 2);
        let parent = this.values[parentIdx];
        if (element.priority >= parent.priority) break;
        this.values[parentIdx] = element;
        this.values[idx] = parent;
        idx = parentIdx;
      }
    }
    dequeue() {
      const min = this.values[0];
      const end = this.values.pop();
      if (this.values.length > 0) {
        this.values[0] = end;
        this.sinkDown();
      }
      return min;
    }
    sinkDown() {
      let idx = 0;
      const length = this.values.length;
      const element = this.values[0];
      while (true) {
        let leftChildIdx = 2 * idx + 1;
        let rightChildIdx = 2 * idx + 2;
        let leftChild, rightChild;
        let swap = null;
  
        if (leftChildIdx < length) {
          leftChild = this.values[leftChildIdx];
          if (leftChild.priority < element.priority) {
            swap = leftChildIdx;
          }
        }
        if (rightChildIdx < length) {
          rightChild = this.values[rightChildIdx];
          if (
            (swap === null && rightChild.priority < element.priority) ||
            (swap !== null && rightChild.priority < leftChild.priority)
          ) {
            swap = rightChildIdx;
          }
        }
        if (swap === null) break;
        this.values[idx] = this.values[swap];
        this.values[swap] = element;
        idx = swap;
      }
    }
  }
  
  //Dijkstra's algorithm only works on a weighted graph.
  
  class WeightedGraph {
    constructor() {
      this.adjacencyList = {};
    }
    addVertex(vertex) {
      if (!this.adjacencyList[vertex]) this.adjacencyList[vertex] = [];
    }
    addEdge(vertex1, vertex2, weight) {
      this.adjacencyList[vertex1].push({ node: vertex2, weight });
      this.adjacencyList[vertex2].push({ node: vertex1, weight });
    }
    Dijkstra(start, finish) {
      const nodes = new PriorityQueue();
      const distances = {};
      const previous = {};
      let path = []; //to return at end
      let smallest;
      //build up initial state
      for (let vertex in this.adjacencyList) {
        if (vertex === start) {
          distances[vertex] = 0;
          nodes.enqueue(vertex, 0);
        } else {
          distances[vertex] = Infinity;
          nodes.enqueue(vertex, Infinity);
        }
        previous[vertex] = null;
      }
      // as long as there is something to visit
      while (nodes.values.length) {
        smallest = nodes.dequeue().val;
        if (smallest === finish) {
          //WE ARE DONE
          //BUILD UP PATH TO RETURN AT END
          while (previous[smallest]) {
            path.push(smallest);
            smallest = previous[smallest];
          }
          break;
        }
        if (smallest || distances[smallest] !== Infinity) {
          for (let neighbor in this.adjacencyList[smallest]) {
            //find neighboring node
            let nextNode = this.adjacencyList[smallest][neighbor];
            //calculate new distance to neighboring node
            let candidate = distances[smallest] + nextNode.weight;
            let nextNeighbor = nextNode.node;
            if (candidate < distances[nextNeighbor]) {
              //updating new smallest distance to neighbor
              distances[nextNeighbor] = candidate;
              //updating previous - How we got to neighbor
              previous[nextNeighbor] = smallest;
              //enqueue in priority queue with new priority
              nodes.enqueue(nextNeighbor, candidate);
            }
          }
        }
      }
      return path.concat(smallest).reverse();
    }
  }

var graph = new WeightedGraph();
graph.addVertex("A");
graph.addVertex("B");
graph.addVertex("C");
graph.addVertex("D");
graph.addVertex("E");
graph.addVertex("F");

graph.addEdge("A", "B", 5);
graph.addEdge("A", "C", 2);
graph.addEdge("B", "D", 4);
graph.addEdge("B", "E", 2);
graph.addEdge("C", "B", 8);
graph.addEdge("C", "E", 7);
graph.addEdge("D", "E", 6);
graph.addEdge("D", "F", 3);
graph.addEdge("E", "F", 1);

console.log(graph.Dijkstra("A", "F"));