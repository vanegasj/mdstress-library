#include <iostream>
#include <cassert>

#include "mds_stressgrid.h"

bool test_one(int input) {
	

	assert(input == 1 && "test_one; Input should evaluate to 1");
	
	if (input == 1) {
		std::cout << "test_example/test_one has passed" << std::endl;
	} else {
		std::cout << "test_example/test_one has failed" << std::endl;
		return false;
	}


	return true;
}

int main() {
	test_one(1);
	test_one(2);
	return 0;
}
// Will probably have to compile like this (but this doesn't work):
// g++ -Iinclude src/*.cpp test/test_mds_stressgrid.cpp -o test_mds_stressgrid

