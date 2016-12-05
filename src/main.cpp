#define MAIN2
#ifdef MAIN2
//#define FIXED_ZERO
#include "network.h"

#include "fstream"
#include "iostream"
#include <string>
#include <sstream>

using namespace mlp;
using namespace std;

#define RETRAIN
FILE *fp;

int main(int argc,char* argv[]){
	vec2d_t train_x;
	vec_t train_y;
	vec2d_t test_x;
	vec_t test_y;
	//std::ofstream ofile("weight.txt");
    fp = fopen(argv[argc-1],"w");
	LOAD_MNIST_TEST(test_x, test_y);
	LOAD_MNIST_TRAIN(train_x, train_y);
    float_t learningRate = atof(argv[1]);
	float_t inWD = atof(argv[2]);
	int lay = atoi(argv[3]);
	Mlp n(learningRate, 0.01, inWD);
	for (int i = 4; i < argc-1; i++)
	{
		printf("%d\n",atoi(argv[i]));
		if (i == 4) n.add_layer(new FullyConnectedLayer(28 *28, atoi(argv[i]), new sigmoid_activation));
		else n.add_layer(new FullyConnectedLayer(atoi(argv[i - 1]),atoi(argv[i]), new sigmoid_activation));
	}
	if (lay != 0)
		n.add_layer(new FullyConnectedLayer(atoi(argv[argc - 2]), 10, new sigmoid_activation));
	else
		n.add_layer(new FullyConnectedLayer(28*28, 10, new sigmoid_activation));
	/*if (lay == 0) n.add_layer(new FullyConnectedLayer(28 *28, 10, new sigmoid_activation));
	else if (lay == 1)
	{
		n.add_layer(new FullyConnectedLayer(28 *28, 100, new sigmoid_activation));
		n.add_layer(new FullyConnectedLayer(100, 10, new sigmoid_activation));
	}*/
	//n.add_layer(new FullyConnectedLayer(28 *28, 10, new sigmoid_activation));
    n.train(train_x,train_y,60000,test_x,test_y,10000);

	
	/*for(int i=5;i<=8;i++){
		std::ifstream weight_file("weight.txt");
		n.fin_weight(weight_file);
		std::stringstream ss;
		ss << "generate_fault" << i << ".log";
		std::cout<<ss.str()<<std::endl;
		std::ifstream generate_logfile(ss.str());
		int j,index;
		for(j=0;j<500;j++)
		{
			float fault_;
			generate_logfile>>index>>fault_;
			if(fault_>0.43+(8-i)*0.05 && fault_<0.46+(8-i)*0.05)break;
			
		}
		std::cout<<j<<' '<<index<<std::endl;
		if (i!=501){
			ss.clear();
			ss.str("");
			ss << "fault" << i << "/fault" << j << ".txt";
			std::ifstream fault_file(ss.str());
			std::cout<<ss.str()<<std::endl;
			n.fin_fault(fault_file);

			ss.clear();
			ss.str("");
			ss << "retrain" << i << "/fault" << j << ".logN30I250";
			std::cout<<"logfile:"<<ss.str()<<std::endl;
			std::ofstream fault_log(ss.str());

			ss.clear();
			ss.str("");
			ss << "retrain" << i << "/fault" << j << ".nwN30I250";
			std::cout << "new weight file:" << ss.str() << std::endl;
			std::ofstream weight_file(ss.str());

			ss.clear();
			ss.str("");
			ss << "retrain" << i << "/fault" << j << ".fwN30I250";
			std::cout << "fixed file:" << ss.str() << std::endl;
			std::ofstream fix_file(ss.str());

			std::streambuf *coutbuf = std::cout.rdbuf();
			std::cout.rdbuf(fault_log.rdbuf());
			n.fault_test(test_x, test_y, 10000);
			n.test(test_x, test_y, 10000);
			for (int i = 0; i < 250; i++){
				n.retrain(train_x, train_y, 60000,0.05);
				n.fault_test(test_x, test_y, 10000);
			}
			n.fout_weight(weight_file);
			n.fout_fixed(fix_file);

			weight_file.close();
			fix_file.close();

			fault_file.close();
			std::cout.rdbuf(coutbuf);
		}
		weight_file.close();
		generate_logfile.close();
		
	}
*/
	return 0;
}
#endif