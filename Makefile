all: dl-optimize

dl-optimize: dl-optimize.cpp
	g++ -std=c++11 -O2 -g3 dl-optimize.cpp -o dl-optimize -fopenmp
