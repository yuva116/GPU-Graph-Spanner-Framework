#include "algorithms/fgv/fgv.hpp"
#include "graph/csr.hpp"
#include "graph/graph_loader.hpp"
#include "gpu/gpu_graph.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    bool has_input_edge(const spanner::CSRGraph& graph,int source,int destination)
    {
        const int begin = static_cast<int>(graph.offsets()[source]);
        const int end = static_cast<int>(graph.offsets()[source + 1]);

        return std::binary_search(graph.neighbors_data().begin() + begin,graph.neighbors_data().begin() + end,destination);
    }

    int spanner_distance(const std::vector<spanner::Edge>& edges,int num_vertices,int source,int target)
    {
        if(source == target)
        {
            return 0;
        }

        std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(num_vertices));

        for(const spanner::Edge& edge : edges)
        {
            adjacency[edge.source].push_back(edge.destination);
            adjacency[edge.destination].push_back(edge.source);
        }

        std::vector<int> distance(static_cast<std::size_t>(num_vertices),-1);
        std::vector<int> queue(static_cast<std::size_t>(num_vertices));

        int head = 0;
        int tail = 0;

        queue[tail++] = source;
        distance[source] = 0;

        while(head < tail)
        {
            const int vertex = queue[head++];

            for(const int neighbor : adjacency[vertex])
            {
                if(distance[neighbor] != -1)
                {
                    continue;
                }

                distance[neighbor] = distance[vertex] + 1;

                if(neighbor == target)
                {
                    return distance[neighbor];
                }

                queue[tail++] = neighbor;
            }
        }

        return -1;
    }
}

int main()
{
    try
    {
        const spanner::GraphData graph_data = spanner::GraphLoader::load_edge_list("datasets/test_graph.txt");
        const spanner::Graph graph(graph_data.num_vertices,graph_data.edges,false);
        const spanner::CSRGraph csr_graph(graph);

        spanner::GPUGraph gpu_graph(csr_graph);

        constexpr int k = 3;
        constexpr int stretch_bound = 2 * k - 1;

        spanner::FGV fgv(gpu_graph,k);
        fgv.run();

        const std::vector<spanner::Edge>& spanner_edges = fgv.edges();

        std::cout << "Vertices: " << csr_graph.num_vertices() << '\n';
        std::cout << "Input edges: " << csr_graph.num_edges() << '\n';
        std::cout << "Spanner edges: " << spanner_edges.size() << '\n';

        if(spanner_edges.empty())
        {
            throw std::runtime_error("FGV produced an empty spanner");
        }

        for(const spanner::Edge& edge : spanner_edges)
        {
            if(edge.source == edge.destination)
            {
                throw std::runtime_error("FGV produced a self-loop");
            }

            if(edge.source < 0 || edge.destination < 0 || edge.source >= static_cast<int>(csr_graph.num_vertices()) || edge.destination >= static_cast<int>(csr_graph.num_vertices()))
            {
                throw std::runtime_error("FGV produced an invalid vertex");
            }

            if(edge.source > edge.destination)
            {
                throw std::runtime_error("FGV edge is not canonicalized");
            }

            if(!has_input_edge(csr_graph,edge.source,edge.destination))
            {
                throw std::runtime_error("FGV produced an edge not present in the input graph");
            }
        }

        for(std::size_t index = 1;index < spanner_edges.size();++index)
        {
            const spanner::Edge& previous = spanner_edges[index - 1];
            const spanner::Edge& current = spanner_edges[index];

            if(previous.source > current.source || (previous.source == current.source && previous.destination >= current.destination))
            {
                throw std::runtime_error("FGV edges are not sorted and unique");
            }
        }

        std::cout << "Structural checks: PASS\n";

        int checked_edges = 0;
        int maximum_stretch = 0;

        for(std::size_t vertex = 0;vertex < csr_graph.num_vertices();++vertex)
        {
            const int begin = static_cast<int>(csr_graph.offsets()[vertex]);
            const int end = static_cast<int>(csr_graph.offsets()[vertex + 1]);

            for(int index = begin;index < end;++index)
            {
                const int neighbor = csr_graph.neighbors_data()[index];

                if(static_cast<int>(vertex) >= neighbor)
                {
                    continue;
                }

                if(vertex == static_cast<std::size_t>(neighbor))
                {
                    continue;
                }

                ++checked_edges;

                const int distance = spanner_distance(spanner_edges,static_cast<int>(csr_graph.num_vertices()),static_cast<int>(vertex),neighbor);

                if(distance < 0)
                {
                    std::cerr << "Disconnected input edge: (" << vertex << "," << neighbor << ")\n";
                    throw std::runtime_error("FGV spanner is disconnected");
                }

                maximum_stretch = std::max(maximum_stretch,distance);

                if(distance > stretch_bound)
                {
                    std::cerr << "Stretch violation: (" << vertex << "," << neighbor << ") -> " << distance << '\n';
                    throw std::runtime_error("FGV stretch bound violated");
                }
            }
        }

        std::cout << "Checked edges: " << checked_edges << '\n';
        std::cout << "Maximum observed stretch: " << maximum_stretch << '\n';
        std::cout << "Required stretch bound: " << stretch_bound << '\n';
        std::cout << "Stretch check: PASS\n";
        std::cout << "FGV TEST: PASS\n";

        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FGV TEST: FAIL\n";
        std::cerr << error.what() << '\n';
        return 1;
    }
}