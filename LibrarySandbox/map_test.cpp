//map_test.cpp

#include <iostream>
#include <map>


int main(){
	
	std::cout << "hello " << std::endl;

	std::map<int, float> coo;

	int a[] = {1,2,5,12,3};
	int b[] = {1,3,14,19,30};

	float A[] = {12.2, 100.45, 3.12, 534.23, 122.};
	float B[] = {61.634, 2.35, 8.12, 4.23, 1.};

	for (int i = 0; i < 5; i++){
		coo[a[i]] += A[i];// + coo[a[i]];
		coo[b[i]] += B[i];// + coo[b[i]];
	}

	std::map<int, float>::iterator it = coo.begin();
    while(it != coo.end())
    {
        std::cout << it->first << " :: " << it->second << std::endl;
        it++;
    }


	return 0;
}


