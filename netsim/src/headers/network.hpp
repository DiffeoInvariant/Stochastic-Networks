#ifndef NETSIM_NETWORK_HPP
#define NETSIM_NETWORK_HPP

#include <petscdmnetwork.h>
#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/metis.hpp>
#include <boost/graph/distributed/graphviz.hpp>
#include <boost/graph/copy.hpp>
#include <vector>
#include <string>
#include <mpi.h>
#include <type_traits>
#include <fstream>


namespace bgl = boost::graph;

namespace netsim
{
	template<typename Directedness, typename VertexProperties,
	       	 typename EdgeProperties, typename List_t=boost::listS >
	using BoostGraph = boost::adjacency_list<List_t, boost::distributedS<bgl::distributed::mpi_process_group, List_t>, Directedness,
											VertexProperties, EdgeProperties>;
	//distance stored in vertices
	template<typename T=double>
	using UndirectedWeightedBoostGraph = BoostGraph<boost::undirectedS, boost::property<boost:vertex_distance_t, T>,
													boost::property<boost::edge_weight_t, T> >;
	//unnamed vertices
	template<typename List_t = boost::listS>
	using UndirectedUnweightedBoostGraph = boost::adjacency_list<List_t, 
												boost::distributedS<bgl::distributed::mpi_process_group, List_t>,
												boost::undirectedS
													>;
	//named vertices
	template<typename List_t = boost::listS>
	using UndirectedUnweightedNamedBoostGraph = BoostGraph<boost::undirectedS,
													boost::property<boost:vertex_name_t, std::string>,
													boost::property<boost::edge_index_t, std::size_t> >;


	enum GraphStorageType
	{
		BoostAdjList,
		PetscDMNet,//makes a PETSc DMNetwork
		BoostAdjMat,
		BoostAdjListMat,//stores both a boost::adjacency_list and a boost::adjacency_matrix
		BoostAdjListPetscDMNet //stores both a boost::adjacency_list and a PETSc DMNetwork
	};

	enum DirectedType
	{
		Directed,
		Undirected
	};

	enum WeightedType
	{
		Weighted,
		Unweighted
	};



	template<
		DirectedType Directedness, WeightedType Weight,
		typename List_t, typename VertexProperties,
	   	typename EdgeProperties, GraphStorageType Storage_t
			>
	class Graph;


	/* specialization for undirected, unweighted graph, stored in both a boost::adjacency_list and a 
	 * boost::adjacency_matrix
	 */
	template<typename List_t, typename VertexProperties, typename EdgeProperties>
	class Graph<Undirected, Unweighted, List_t, VertexProperties, EdgeProperties, BoostAdjListMat>
	{
	public:

		using AdjList_t = 		boost::adjacency_list<List_t, 
		 boost::distributedS<bgl::distributed::mpi_process_group, List_t>,
		 boost::undirectedS, VertexProperties, EdgeProperties
			 >;

		using AdjMat_t = boost::adjacency_matrix<boost::undirectedS>;


		//construct graph from METIS input file
		Graph(std::string filename) 
		{
			std::ifstream in(filename);

			bgl::metis_reader reader(in);
		
			//read in adjacency list
			m_adj_list = AdjList_t(reader.begin(), reader.end(), reader.weight_begin(),
								   reader.num_vertices());

			m_adj_mat = AdjMat_t(boost::num_vertices(m_adj_list));
			//construct matrix
			boost::copy_graph(m_adj_list, m_adj_mat);
		};


	private:

		AdjList_t	m_adj_list:

		AdjMat_t 	m_adj_mat;

	};

	/*Specialization for PETSc DMNetwork to hold data*/
	template<typename VertexProperties=void, typename EdgeProperties=void>
	class Graph<Undirected, Unweighted, void, VertexProperties, EdgeProperties, PetscDMNet>
	{
	public:

		Graph(



	




}//end namespace netsim
#endif //NETSIM_NETWORK_HPP
