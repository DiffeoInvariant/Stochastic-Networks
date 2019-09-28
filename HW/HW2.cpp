#include "../netsim/include/io.hpp"

//#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edge_list.hpp>
#include <boost/graph/use_mpi.hpp>
#include <boost/config.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/distributed/connected_components_parallel_search.hpp>
#include <boost/graph/random.hpp>
#include <boost/property_map/parallel/distributed_property_map.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/parallel/distribution.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/graph/distributed/graphviz.hpp>
//#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/random.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <ctime>
#include <algorithm> 
#include <list>

namespace pbgl = boost::graph::distributed;

namespace mpi = boost::mpi;

using Graph = boost::adjacency_list<boost::listS,
								    boost::distributedS<pbgl::mpi_process_group, boost::vecS>,
							   		boost::undirectedS>;


//using ERGen = boost::erdos_renyi_iterator<boost::minstd_rand, Graph>;

void random_erdos_renyi_graph(Graph& g, pbgl::mpi_process_group& pg, int rank, int n, double k, int fast_response_pct=50)
{
	boost::minstd_rand gen, response_gen;
	int seed = std::time(0);
	gen.seed(seed);


	//broadcast seed from root
	pbgl::broadcast(pg, seed, 0);
	
	response_gen.seed(1);

	double p = (k)/((double)n);

	if(rank == 0)
	{	//add all the edges from root

		std::cout << "Adding edges from root...\n";

		for(boost::erdos_renyi_iterator<boost::minstd_rand, Graph> first(gen, n, p), end; first != end; ++first)
		{
			Graph::lazy_add_edge lazy_adder = boost::add_edge(boost::vertex(first->first, g),
															  boost::vertex(first->second, g),
															  g);
			if(static_cast<int>(response_gen()) % 100 < fast_response_pct){
				//make this edge readily accessible
				std::pair<boost::graph_traits<Graph>::edge_descriptor, bool> res(lazy_adder);
				if(not (boost::source(res.first, g) == boost::vertex(first->first, g)) or not
					   (boost::source(res.first, g) == boost::vertex(first->second, g)) )
				{
					std::cerr << "Some error happened.\n";
				}
			}
		}

		std::cout << "Done adding edges. Distributing and synchronizing the graph..." << std::endl;
	}
	
	boost::synchronize(g);
	
	std::cout << "Done." << std::endl;
}

std::pair<int, std::vector<int>> connected_components(Graph& g)
{
	std::vector<int> components(boost::num_vertices(g));
	int num_comp = pbgl::connected_components(g, &components[0]);
	return std::make_pair(num_comp, components);
}

std::vector<int> component_sizes(const std::vector<int>& components, int num_components)
{
	/* this returns a vector whose i-th entry is the number of vertices in component i of the graph
	 */
	std::vector<int> comp_count(num_components);

	std::fill(comp_count.begin(), comp_count.end(), 0);
	for(const auto& v : components)
	{
		comp_count[v]++;
	}
	return comp_count;
}

int main(int argc, char** argv)
{
	mpi::environment env(argc, argv);

	pbgl::mpi_process_group pg;
	int rank = pbgl::process_id(pg);
	int nproc = pbgl::num_processes(pg);

	// problem 4a
	int n = 500;
	double k = 3.0;

	boost::parallel::variant_distribution<pbgl::mpi_process_group> dist500 = boost::parallel::block(pg, n);

	boost::minstd_rand gen;

	gen.seed(std::time(0));

	Graph er_graph500(boost::erdos_renyi_iterator<boost::minstd_rand, Graph>(gen, n, k/((double)n)),
					  boost::erdos_renyi_iterator<boost::minstd_rand, Graph>(), 
					  n, pg, dist500);

	boost::synchronize(er_graph500);

	std::vector<int> local_cc_vec(boost::num_vertices(er_graph500));
	
	using ComponentMap = boost::iterator_property_map<std::vector<int>::iterator,
													 boost::property_map<Graph, 
													 boost::vertex_index_t>::type>;

	ComponentMap comp(local_cc_vec.begin(), boost::get(boost::vertex_index, er_graph500));

	//do a parallel search for the connected components
	int ncomp = pbgl::connected_components_ps(er_graph500, comp);


	std::cout << "There are x " << ncomp << " connected components.\n";

	//problem 4b
	n = 1E4;

	std::list<double> S_list;


	for(double i = 0; i <= 100; ++i){
		Graph g;
		random_erdos_renyi_graph(g, pg, rank, n, 0.1 + 2.4 * i / 100);
		auto nV = boost::num_vertices(g);
		auto ccg = connected_components(g);
		auto cc_sizesg = component_sizes(ccg.second, ccg.first);

		auto max_cc_sizeg = std::max_element(cc_sizesg.begin(), cc_sizesg.end());

		S_list.push_back(((double)(*max_cc_sizeg))/((double)nV));
	}

	return 0;
}




