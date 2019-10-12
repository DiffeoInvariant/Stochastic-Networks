#ifndef NETSIM_IO_HPP
#define NETSIM_IO_HPP

#include <boost/spirit/include/qi.hpp>
#include <boost/graph/edge_list.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include <iostream>


namespace netsim
{
	inline namespace io
	{
		using IOEdge = std::pair<int, int>;
		using IOEdgeList = std::vector<IOEdge>;
		using IOGraph = boost::edge_list<IOEdgeList::iterator>;
	//	using DirectedIOMat = boost::adjacency_matrix<boost::directedS>;

		namespace qi = boost::spirit::qi;

		template<typename T>
		T read_graph_from_txt(std::string filename, bool print_status=true){
			std::ifstream indata(filename);

			indata >> std::noskipws;

			boost::spirit::istream_iterator fl(indata), ln;

			IOEdgeList edges;

			bool parse_success = qi::phrase_parse(fl, ln,
												  (qi::int_ >> ',' >> qi::int_) % qi::eol,
											   	  qi::blank, edges);

			T G(edges.begin(), edges.end());
	
			if(print_status){
				if(parse_success){
					std::cout << "Successfully parsed file " << filename << " and read in graph with " 
							  << boost::num_edges(G) << " edges.\n";
				} else{
					std::cerr << "Parser error.\n";
				}

				if(fl != ln) {
					std::cerr << "Remaining unparsed lines: \n '" << std::string(fl, ln) << "'\n";
				}
			}

			return G;
		}


		IOGraph read_graph_list_from_txt(std::string filename, bool print_status=true){
			return read_graph_from_txt<IOGraph>(filename, print_status);
		}

	//	DirectedIOMat read_graph_directed_mat_from_txt(std::string filename, bool print_status=true){
	//		return read_graph_from_txt<DirectedIOMat>(filename, print_status);
	//	}




	}
}
#endif
