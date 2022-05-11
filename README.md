We present an algorithm for solving two-player safety games that combines a mixed forward/backward search strategy with a symbolic representation of the state space.
By combining forward and backward exploration, our algorithm can synthesize strategies that are eager in the sense that they try to prevent progress towards the error states as soon as possible, whereas standard backwards algorithms often produce permissive solutions that only react when absolutely necessary.
We provide experimental results for two new sets of benchmarks, as well as the benchmark set of the Reactive Synthesis Competition (SYNTCOMP) 2017.
The results show that our algorithm in many cases produces more eager strategies than a standard backwards algorithm, and solves a number of benchmarks that are intractable for existing tools.
Finally, we observe a connection between our algorithm and a recently proposed algorithm for the synthesis of controllers that are robust against disturbances, pointing to possible future applications.

Paper: 

Jacobs, S., Sakr, M.: A symbolic algorithm for lazy synthesis of eager strategies. 
Acta Informatica 57 (1), 81-106.


Sample command to run the tool:

LazySafetySynt.py inoutfile.aag -o outputfile.aag
