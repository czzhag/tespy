import numpy as np

def gettemplists():
	templists = {}
	templists['20180817'] = [252,253,257,265,278,290,305,321,337,355,373,390,407,427,433,445,454,470,490]
	templists['20190409'] = [248, 252, 261, 276, 294, 314, 338, 360, 383, 408, 430, 458, 481, 504]
	templists['20190410'] = [261, 293, 295, 302, 313, 327, 344, 363, 383, 404]	
	templists['20190412'] = [289, 292, 300, 307, 313,
				 325, 329, 346, 366, 388,
				 409, 430, 461, 465]
	templists['20190528'] = [265, 270, 288, 314, 345,
				 379]
	templists['20190529'] = [264,271,
				314,345,378,
				412,446]
	#templists['20190811'] = [282,299,308,317,326,338,350,367,371,410,443,471]
	templists['20190730'] = [259,265,276,291,310,330,352,376,399,427,463]
	templists['20190811'] = [282,299,308,317,326,338,350,410,443,471]
	templists['20190908'] = [304,321,340,360,380,401,421,442,464,486,508]
	templists['20191109'] = [321, 331, 343, 358, 376, 395, 415, 436]
	templists['20191116'] = [264, 266, 277, 292, 311, 331, 353, 376,
				399, 421, 444, 474, 508]
	return templists
